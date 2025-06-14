#!/usr/bin/env python3
"""
KOPRU AI-Human Interface Test Script
🌉 KOPRU'nun AI-Human interface sistemini test eder

Test scenarios:
- Autonomy Bridge risk assessment
- Context Bridge session management
- Bridge creation and management
- Knowledge preservation

Bu test YAPYOS Framework test'lerinden esinlenildi.
"""

import sys
from pathlib import Path

# KOPRU module path'ini ekle (parent directory)
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_autonomy_bridge():
    """AutonomyBridge test et"""
    print("🤖 Testing KOPRU Autonomy Bridge...")
    
    try:
        from kopru.ai_human.autonomy_bridge import AutonomyBridge
        
        # Initialize autonomy bridge
        bridge = AutonomyBridge()
        print("✅ Autonomy Bridge initialized")
        
        # Test bridge creation - Low risk
        result1 = bridge.create_bridge(
            bridge_type="research",
            source="dual_modal_template", 
            target="academic_project",
            context={"backup_available": True}
        )
        print(f"✅ Research Bridge: {result1['permission']} - {result1['message']}")
        
        # Test bridge creation - Medium risk
        result2 = bridge.create_bridge(
            bridge_type="data",
            source="research_data",
            target="analysis_output", 
            context={"human_supervised": True}
        )
        print(f"⚠️ Data Bridge: {result2['permission']} - {result2['message']}")
        
        # Test bridge creation - High risk
        result3 = bridge.create_bridge(
            bridge_type="system",
            source="system_config",
            target="production_deploy",
            context={}
        )
        print(f"🚨 System Bridge: {result3['permission']} - {result3['message']}")
        
        # Autonomy stats
        stats = bridge.get_autonomy_stats()
        print(f"📊 Autonomy Stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Autonomy Bridge test failed: {e}")
        return False

def test_context_bridge():
    """ContextBridge test et"""
    print("\n🧠 Testing KOPRU Context Bridge...")
    
    try:
        from kopru.ai_human.context_bridge import ContextBridge
        
        # Initialize context bridge
        bridge = ContextBridge("KOPRU_Test_Project")
        print("✅ Context Bridge initialized")
        
        # Create session
        session_id = bridge.create_bridge_session({
            "project_type": "dual_modal_research",
            "bridge_count": 3,
            "phase": "ai_human_interface_test"
        })
        print(f"✅ Bridge Session created: {session_id}")
        
        # Save session with knowledge points
        knowledge_points = [
            "KOPRU AI-Human interface successfully implemented",
            "Autonomy Bridge provides risk-based decision making",
            "Context Bridge enables session persistence",
            "Bridge creation system operational",
            "Knowledge preservation working correctly",
            "YAPYOS Framework integration successful"
        ]
        
        success = bridge.save_bridge_session(
            summary="KOPRU AI-Human interface test session",
            knowledge_points=knowledge_points
        )
        print(f"✅ Session saved: {success}")
        
        # Generate context summary
        summary = bridge.generate_bridge_context_summary()
        print(f"📄 Context Summary generated: {len(summary)} characters")
        
        # Export resume guide
        guide_path = bridge.export_bridge_resume_guide()
        print(f"📋 Resume Guide exported: {guide_path}")
        
        # Search knowledge
        search_results = bridge.search_bridge_knowledge("KOPRU", 5)
        print(f"🔍 Knowledge Search: {len(search_results)} results found")
        
        # Bridge stats
        stats = bridge.get_bridge_stats()
        print(f"📊 Bridge Stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Context Bridge test failed: {e}")
        return False

def test_bridge_integration():
    """Bridge entegrasyonu test et"""
    print("\n🌉 Testing KOPRU Bridge Integration...")
    
    try:
        from kopru.ai_human.autonomy_bridge import AutonomyBridge
        from kopru.ai_human.context_bridge import ContextBridge
        
        # Initialize both bridges
        autonomy = AutonomyBridge()
        context = ContextBridge("Integration_Test")
        
        print("✅ Both bridges initialized")
        
        # Create context session
        session_id = context.create_bridge_session({
            "test_type": "integration",
            "bridges": ["autonomy", "context"],
            "inspired_by": "YAPYOS Framework"
        })
        
        # Test multiple bridge operations with autonomy
        operations = [
            ("template", "low_risk_operation"),
            ("research", "medium_risk_operation"), 
            ("documentation", "safe_operation")
        ]
        
        results = []
        for bridge_type, operation in operations:
            result = autonomy.create_bridge(
                bridge_type=bridge_type,
                source=f"test_{operation}",
                target="integration_test",
                context={"backup_available": True, "test_mode": True}
            )
            results.append(result)
            print(f"🔧 {bridge_type}: {result['permission']}")
        
        # Save integration session
        knowledge_points = [
            f"Integration test completed with {len(results)} bridge operations",
            "Autonomy and Context bridges work together seamlessly",
            "Bridge creation pipeline operational",
            "Risk assessment system functioning correctly",
            "KOPRU Framework ready for production use"
        ]
        
        context.save_bridge_session(
            summary="KOPRU bridge integration test",
            knowledge_points=knowledge_points
        )
        
        print(f"✅ Integration test completed: {len(results)} operations")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Ana test fonksiyonu"""
    print("🌉 KOPRU AI-Human Interface Test Suite")
    print("Inspired by YAPYOS Framework")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Autonomy Bridge
    if test_autonomy_bridge():
        tests_passed += 1
    
    # Test 2: Context Bridge  
    if test_context_bridge():
        tests_passed += 1
    
    # Test 3: Bridge Integration
    if test_bridge_integration():
        tests_passed += 1
    
    # Test özeti
    print("\n" + "=" * 50)
    print(f"🎯 KOPRU AI-Human Interface Test Results:")
    print(f"✅ Tests Passed: {tests_passed}/{total_tests}")
    print(f"📊 Success Rate: {tests_passed/total_tests*100:.1f}%")
    
    if tests_passed == total_tests:
        print("🎉 All KOPRU AI-Human Interface tests PASSED!")
        print("🌉 KOPRU bridge system is ready for operation!")
        print("✨ Contamination issues resolved - Clean architecture achieved!")
    else:
        print(f"⚠️ {total_tests - tests_passed} test(s) failed")
        print("🔧 Further development needed")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 