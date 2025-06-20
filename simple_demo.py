from drug_discovery_pipeline import DrugDiscoveryPipeline

def simple_demo():
    print("🧬 SIMPLE DRUG DISCOVERY DEMO")
    print("=" * 40)
    
    molecules = ['CC(=O)Oc1ccccc1C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'CCO']
    
    pipeline = DrugDiscoveryPipeline("Simple_Demo")
    pipeline.load_compounds(molecules)
    pipeline.calculate_descriptors()
    pipeline.filter_drug_like('lipinski')
    
    # Show results
    df = pipeline.descriptors_df
    for i, row in df.iterrows():
        drug_like = "✅ Drug-like" if row['Drug_Like'] else "Not drug-like"
        print(f"   {row['Name']}: {drug_like} (MW: {row['MolWt']:.1f})")
    
    print("\n✅ Demo Discovery complete!")

if __name__ == "__main__":
    simple_demo()