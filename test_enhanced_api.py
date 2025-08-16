#!/usr/bin/env python3
"""
Test script for ARIA Vision Pro Enhanced API
Demonstrates the depth-enhanced vision chat capabilities
"""

import requests
import json
import time

# ARIA Vision Pro endpoint
ARIA_URL = "http://localhost:5000"

def test_enhanced_chat():
    """Test the enhanced /v1/chat/completions endpoint"""
    
    print("🔥 Testing ARIA Vision Pro Enhanced Chat API")
    print("=" * 50)
    
    # Test queries that showcase depth intelligence
    test_queries = [
        "What do you see in front of the camera?",
        "How far away are the objects in the scene?", 
        "Are there any objects close to the camera?",
        "Describe the spatial layout of the scene",
        "What's the closest object and what's the furthest?",
        "Can you tell me about the depth of field in this scene?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🤖 Query {i}: {query}")
        print("-" * 40)
        
        # Send request to enhanced API
        response = requests.post(f"{ARIA_URL}/v1/chat/completions", json={
            "max_tokens": 200,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query}
                    ]
                }
            ]
        })
        
        if response.status_code == 200:
            data = response.json()
            if "choices" in data and data["choices"]:
                answer = data["choices"][0]["message"]["content"]
                print(f"✅ ARIA Response: {answer}")
            else:
                print(f"❌ Error: {data}")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
        
        # Small delay between requests
        time.sleep(2)
    
    print("\n" + "=" * 50)
    print("🎯 Enhanced API Test Complete!")
    print("\nKey Benefits Demonstrated:")
    print("• Depth-aware scene understanding")
    print("• Spatial relationship analysis") 
    print("• Enhanced context from depth + vision")
    print("• OpenAI API compatibility")

def check_system_status():
    """Check if ARIA Vision Pro is running and ready"""
    try:
        response = requests.get(f"{ARIA_URL}/status")
        if response.status_code == 200:
            status = response.json()
            print(f"📊 System Status:")
            print(f"   • Depth Model: {'✅ Loaded' if status['model_loaded'] else '❌ Not Ready'}")
            print(f"   • Vision Model: {'✅ Loaded' if status['vision_model_loaded'] else '❌ Not Ready'}")
            print(f"   • Camera: {'🎥 Running' if status['camera_running'] else '📴 Stopped'}")
            return status['model_loaded'] and status['vision_model_loaded']
        else:
            print("❌ Failed to connect to ARIA Vision Pro")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ ARIA Vision Pro not running. Please start web_app_vision.py first!")
        return False

if __name__ == "__main__":
    print("🚀 ARIA Vision Pro Enhanced API Test")
    print("Make sure web_app_vision.py is running first!\n")
    
    if check_system_status():
        print("\n✅ System Ready! Starting enhanced chat test...\n")
        test_enhanced_api()
    else:
        print("\n❌ System not ready. Please:")
        print("   1. Run: python web_app_vision.py")
        print("   2. Click 'Launch Vision' in the web interface")
        print("   3. Wait for models to load")
        print("   4. Run this test script again")
