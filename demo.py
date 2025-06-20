#!/usr/bin/env python3
"""
Drug Discovery Pipeline - Quick Demo
"""

from drug_discovery_pipeline import DrugDiscoveryPipeline
from rdkit import Chem

def quick_demo():
    """Short demo of the drug discovery pipeline"""
    
    print("🧬 QUICK DRUG DISCOVERY DEMO")
    print("=" * 40)
    
    molecules = [
        ('CC(=O)Oc1ccccc1C(=O)O', 'Aspirin'),         
        ('CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'Caffeine'),  
        ('CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O', 'Ibuprofen'), 
        ('CCO', 'Ethanol'),                           
        ('c1ccccc1', 'Benzene'),
        ('CC(C)O', 'Isopropanol'),
        ('CCCCO', 'Butanol'),
        ('c1ccc(cc1)O', 'Phenol'),
        ('CC(=O)O', 'Acetic_acid'),
    ]
    
    smiles_list = [mol[0] for mol in molecules]
    
    # More activity data
    activities = {
        'CC(=O)Oc1ccccc1C(=O)O': 6.5,        
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C': 5.2, 
        'CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O': 7.1, 
        'CCO': 3.0,                           
        'c1ccccc1': 2.1,                      
        'CC(C)O': 3.5,                        
        'CCCCO': 4.2,                         
        'c1ccc(cc1)O': 5.8,                   
        'CC(=O)O': 3.8,                       
    }
    
    # Initialize pipeline
    pipeline = DrugDiscoveryPipeline("Quick_Demo")
    
    # Run core analysis
    pipeline.load_compounds(smiles_list)
    pipeline.calculate_descriptors()
    pipeline.filter_drug_like('lipinski')
    pipeline.generate_fingerprints()
    
    # Virtual screening
    aspirin = Chem.MolFromSmiles('CC(=O)Oc1ccccc1C(=O)O')
    similar = pipeline.virtual_screening(aspirin, similarity_threshold=0.3)
    
    print(f"\n🔍 Found {len(similar)} molecules similar to aspirin")
    
    # Build prediction model
    qsar_results = pipeline.build_qsar_model(activities)
    if qsar_results:
        print(f"🎯 Model R² Score: {qsar_results['test_r2']:.3f}")
        
        # Top important features
        top_features = sorted(qsar_results['feature_importance'].items(), 
                             key=lambda x: x[1], reverse=True)[:5]
        print("📈 Top 5 Important Features:")
        for feature, importance in top_features:
            print(f"   • {feature}: {importance:.3f}")
    
    # Generate report
    pipeline.generate_comprehensive_report()
    
    print("\n✅ Demo complete!")
    print("Files created:")
    print("- Quick_Demo_report.txt")
    print("\n🎓 Analysis successful! Check the report for detailed results.")

if __name__ == "__main__":
    quick_demo()