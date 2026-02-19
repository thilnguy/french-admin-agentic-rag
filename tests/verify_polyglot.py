
import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from skills.admin_translator import translate_admin_text
from src.utils.logger import logger

async def verify_polyglot():
    print("🌍 STARTING POLYGLOT VERIFICATION STRESS TEST 🌍")
    print("==================================================")
    
    test_cases = [
        {
            "scenario": "ENGLISH (Standard)",
            "input_text": "**[DONNER]**: Pour obtenir un passeport, vous devez vous rendre à la Mairie avec votre carte d'identité.",
            "target_lang": "English",
            "expected_terms": ["Passeport", "Mairie"],
            "expected_tags": ["[GIVE]"],
            "forbidden_tags": ["[DONNER]", "[CUNG CẤP]"]
        },
        {
            "scenario": "VIETNAMESE (Complex)",
            "input_text": "**[EXPLIQUER]**: Pour le renouvellement de votre Titre de séjour, la Préfecture exige une demande sur l'ANTS.",
            "target_lang": "Vietnamese",
            "expected_terms": ["Titre de séjour", "Préfecture", "ANTS"],
            "expected_tags": ["[GIẢI THÍCH]"],
            "forbidden_tags": ["[EXPLIQUER]", "[EXPLAIN]"]
        },
        {
            "scenario": "FRENCH (Native)",
            "input_text": "**[DEMANDER]**: Avez-vous votre numéro NEPH ?",
            "target_lang": "French",
            "expected_terms": ["NEPH"],
            "expected_tags": ["[DEMANDER]"], # Should remain unchanged or be mirrored
            "forbidden_tags": ["[ASK]", "[HỎI]"]
        }
    ]
    
    failures = []
    
    for case in test_cases:
        print(f"\nTesting Scenario: {case['scenario']}...")
        print(f"Input: {case['input_text']}")
        
        # Determine language for translator (French doesn't really need translation but let's see how it handles it if passed)
        # Actually, for French, the Orchestrator skips translation. But we want to test the REWRITER logic if we were to force it.
        # However, `admin_translator` is usually called for NON-French.
        # Let's assume we want to verify it doesn't break if called, or we skip calling it for French in the real app.
        # But the Requirement was "Scenario 3: French -> French response + NEPH/ANTS preserved".
        # If the Orchestrator skips translation, this is trivial.
        # If we invoke the translator with target="French", it should just return the text or clean it up.
        
        if case['target_lang'] == "French":
             # Special case: The translator prompt is "translate strictly into {target}". 
             # If target is French, it might just output the same.
             result = await translate_admin_text(case['input_text'], "French")
        else:
             result = await translate_admin_text(case['input_text'], case['target_lang'])
             
        print(f"Output: {result}")
        
        # Verification
        case_errors = []
        
        # Check Expected Terms (Glossary)
        for term in case['expected_terms']:
            if term not in result:
                case_errors.append(f"❌ Missing Whitelisted Term: '{term}'")
        
        # Check Expected Tags
        for tag in case['expected_tags']:
            if tag not in result:
                 # Special handling for French tags which might not change
                 if case['target_lang'] == "French" and "[DEMANDER]" in case['input_text'] and "[DEMANDER]" in result:
                     pass
                 else:
                    case_errors.append(f"❌ Missing Target Tag: '{tag}'")
                    
        # Check Forbidden Tags
        for tag in case['forbidden_tags']:
            if tag in result:
                case_errors.append(f"❌ Found Forbidden Tag: '{tag}'")
                
        if case_errors:
            print("FAILED:")
            for err in case_errors:
                print(err)
            failures.append(case['scenario'])
        else:
            print("✅ PASSED")
            
    print("\n==================================================")
    if failures:
        print(f"❌ {len(failures)}/3 Scenarios FAILED")
        sys.exit(1)
    else:
        print("✅ ALL SCENARIOS PASSED - POLYGLOT CERTIFIED")
        sys.exit(0)

if __name__ == "__main__":
    asyncio.run(verify_polyglot())
