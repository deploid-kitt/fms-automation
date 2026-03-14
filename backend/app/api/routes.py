"""FastAPI routes for FMS automation API."""
import os
import uuid
import json
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import aiofiles

from app.core.config import get_settings
from app.models.schemas import (
    UploadResponse, StatusResponse, FMSReport, JobStatus
)
from app.services.video_processor import get_video_processor
from app.services.report_generator import get_report_generator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["fms"])

# In-memory job storage (use Redis in production)
jobs: dict = {}


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@router.post("/upload", response_model=UploadResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a video for FMS analysis.
    
    Returns a job_id for tracking processing status.
    """
    settings = get_settings()
    
    # Validate file type
    allowed_types = ["video/mp4", "video/quicktime", "video/x-msvideo", "video/webm"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}"
        )
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Create upload directory
    upload_dir = settings.upload_dir / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded file
    file_path = upload_dir / file.filename
    
    try:
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            
            # Check file size
            size_mb = len(content) / (1024 * 1024)
            if size_mb > settings.max_video_size_mb:
                raise HTTPException(
                    status_code=400,
                    detail=f"File too large. Maximum: {settings.max_video_size_mb}MB"
                )
            
            await f.write(content)
    except Exception as e:
        logger.error(f"Error saving upload: {e}")
        raise HTTPException(status_code=500, detail="Error saving file")
    
    # Initialize job status
    jobs[job_id] = {
        "status": JobStatus.PENDING,
        "progress": 0,
        "message": "Video uploaded, queued for processing",
        "file_path": str(file_path),
        "created_at": datetime.utcnow().isoformat(),
        "report": None,
        "error": None
    }
    
    # Start background processing
    background_tasks.add_task(process_video_task, job_id, file_path)
    
    logger.info(f"Video uploaded: {file.filename} (job: {job_id})")
    
    return UploadResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message="Video uploaded successfully. Processing started."
    )


async def process_video_task(job_id: str, file_path: Path):
    """Background task to process uploaded video."""
    try:
        jobs[job_id]["status"] = JobStatus.PROCESSING
        jobs[job_id]["message"] = "Processing video..."
        
        processor = get_video_processor()
        
        def progress_callback(progress: float, status):
            jobs[job_id]["progress"] = progress
            if isinstance(status, JobStatus):
                jobs[job_id]["status"] = status
            jobs[job_id]["message"] = str(status)
        
        # Process video
        report = processor.process_video(file_path, job_id, progress_callback)
        
        # Save report
        settings = get_settings()
        report_path = settings.upload_dir / job_id / "report.json"
        async with aiofiles.open(report_path, 'w') as f:
            await f.write(report.model_dump_json(indent=2))
        
        jobs[job_id]["status"] = JobStatus.COMPLETED
        jobs[job_id]["progress"] = 100
        jobs[job_id]["message"] = "Processing complete"
        jobs[job_id]["report"] = report.model_dump()
        
        logger.info(f"Job {job_id} completed. Score: {report.total_score}/21")
        
    except Exception as e:
        logger.exception(f"Error processing job {job_id}")
        jobs[job_id]["status"] = JobStatus.FAILED
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["message"] = f"Processing failed: {str(e)}"


@router.get("/status/{job_id}", response_model=StatusResponse)
async def get_status(job_id: str):
    """Get processing status for a job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return StatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        message=job["message"],
        error=job.get("error")
    )


@router.get("/report/{job_id}", response_model=FMSReport)
async def get_report(job_id: str):
    """Get FMS report for a completed job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Status: {job['status']}"
        )
    
    if not job.get("report"):
        raise HTTPException(status_code=404, detail="Report not found")
    
    return FMSReport(**job["report"])


@router.get("/report/{job_id}/pdf")
async def get_report_pdf(job_id: str):
    """Download PDF report for a completed job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Status: {job['status']}"
        )
    
    if not job.get("report"):
        raise HTTPException(status_code=404, detail="Report not found")
    
    # Generate PDF
    report = FMSReport(**job["report"])
    generator = get_report_generator()
    pdf_bytes = generator.generate_pdf(report)
    
    # Return as download
    return StreamingResponse(
        iter([pdf_bytes]),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=fms_report_{job_id[:8]}.pdf"
        }
    )


@router.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its associated files."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    settings = get_settings()
    job_dir = settings.upload_dir / job_id
    
    # Delete files
    if job_dir.exists():
        import shutil
        shutil.rmtree(job_dir)
    
    # Remove from memory
    del jobs[job_id]
    
    return {"message": "Job deleted"}


@router.get("/jobs")
async def list_jobs():
    """List all jobs (admin endpoint)."""
    return {
        job_id: {
            "status": job["status"],
            "progress": job["progress"],
            "created_at": job.get("created_at"),
            "has_report": job.get("report") is not None
        }
        for job_id, job in jobs.items()
    }
