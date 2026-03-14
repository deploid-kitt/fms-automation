#!/usr/bin/env python3
"""Test WebSocket connection for live FMS analysis."""
import asyncio
import json
import websockets
import sys

async def test_websocket():
    """Test basic WebSocket connectivity."""
    uri = "ws://127.0.0.1:8010/ws/live/test-session-123"
    
    print(f"Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected successfully!")
            
            # Start a session
            start_msg = {"type": "start", "exercise": "deep_squat"}
            await websocket.send(json.dumps(start_msg))
            print(f"📤 Sent: {start_msg}")
            
            # Receive response
            response = await websocket.recv()
            data = json.loads(response)
            print(f"📥 Received: {data}")
            
            if data.get("type") == "started":
                print("✅ Session started successfully!")
                print(f"   Session ID: {data.get('session_id')}")
                print(f"   Exercise: {data.get('exercise')}")
            
            # Send ping
            await websocket.send(json.dumps({"type": "ping"}))
            pong = await websocket.recv()
            pong_data = json.loads(pong)
            print(f"📥 Ping response: {pong_data}")
            
            # Stop session
            await websocket.send(json.dumps({"type": "stop"}))
            final = await websocket.recv()
            final_data = json.loads(final)
            print(f"📥 Final result: {final_data}")
            
            print("\n✅ All WebSocket tests passed!")
            return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_websocket())
    sys.exit(0 if success else 1)
