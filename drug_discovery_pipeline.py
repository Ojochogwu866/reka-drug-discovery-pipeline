#!/usr/bin/env python3
"""
Drug Discovery Pipeline - Core Engine (Updated & Fixed)
======================================================

Author: [Your Name]
Date: [Today's Date]
Course: Introduction to Drug Discovery

A complete pipeline for virtual drug discovery using RDKit.
This is the main engine - import this into other files to use it.

Fixed Issues:
- Morgan fingerprint deprecation warning
- FractionCSP3 descriptor naming
- Pie chart labeling errors
- Better error handling
"""

import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Crippen, Lipinski, AllChem, Draw, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class DrugDiscoveryPipeline:
    """
    Complete drug discovery pipeline using RDKit
    
    This class represents a virtual laboratory for drug discovery!
    """
    
    def __init__(self, project_name: str = "DrugDiscovery_Project"):
        """Initialize our molecular laboratory"""
        self.project_name = project_name
        self.compounds = []
        self.descriptors_df = None
        self.fingerprints = []
        self.qsar_model = None
        self.drug_like_compounds = []
        
    def load_compounds(self, source: str, source_type: str = 'smiles') -> int:
        """Load molecules into our laboratory"""
        print("📦 Loading compounds...")
        self.compounds = []
        
        if source_type == 'smiles':
            if isinstance(source, list):
                for i, smi in enumerate(source):
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        mol.SetProp('_Name', f'Compound_{i+1}')
                        self.compounds.append(mol)
                    else:
                        print(f"   ⚠️  Invalid SMILES: {smi}")
            else:
                with open(source, 'r') as f:
                    for i, line in enumerate(f):
                        parts = line.strip().split('\t')
                        mol = Chem.MolFromSmiles(parts[0])
                        if mol:
                            name = parts[1] if len(parts) > 1 else f'Compound_{i+1}'
                            mol.SetProp('_Name', name)
                            self.compounds.append(mol)
                            
        elif source_type == 'sdf':
            suppl = Chem.SDMolSupplier(source)
            self.compounds = [mol for mol in suppl if mol is not None]
            
        print(f"   ✅ Successfully loaded {len(self.compounds)} valid compounds")
        return len(self.compounds)
    
    def calculate_descriptors(self) -> pd.DataFrame:
        """Calculate molecular properties"""
        print("📏 Calculating molecular descriptors...")
        descriptor_data = []
        
        descriptor_functions = {
            'MolWt': Descriptors.MolWt,
            'LogP': Descriptors.MolLogP,
            'NumHDonors': Descriptors.NumHDonors,
            'NumHAcceptors': Descriptors.NumHAcceptors,
            'TPSA': Descriptors.TPSA,
            'NumRotatableBonds': Descriptors.NumRotatableBonds,
            'NumAromaticRings': Descriptors.NumAromaticRings,
            'NumAliphaticRings': Descriptors.NumAliphaticRings,
            'NumHeteroatoms': Descriptors.NumHeteroatoms,
            'FractionCSP3': Descriptors.FractionCSP3,
            'HeavyAtomCount': Descriptors.HeavyAtomCount,
            'RingCount': Descriptors.RingCount,
            'MolMR': Descriptors.MolMR,
            'BalabanJ': Descriptors.BalabanJ,
            'BertzCT': Descriptors.BertzCT
        }
        
        for i, mol in enumerate(self.compounds):
            row = {
                'CompoundID': i, 
                'Name': mol.GetProp('_Name'),
                'SMILES': Chem.MolToSmiles(mol)
            }
            
            for desc_name, desc_func in descriptor_functions.items():
                try:
                    row[desc_name] = desc_func(mol)
                except:
                    row[desc_name] = np.nan
            
            row.update(self._check_drug_likeness(mol))
            
            descriptor_data.append(row)
        
        self.descriptors_df = pd.DataFrame(descriptor_data)
        print(f"   ✅ Calculated {len(descriptor_functions)} descriptors for each compound")
        return self.descriptors_df
    
    def _check_drug_likeness(self, mol) -> Dict:
        """Check if molecule follows drug-like rules"""
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        tpsa = Descriptors.TPSA(mol)
        rotatable = Descriptors.NumRotatableBonds(mol)
        
        lipinski_violations = sum([
            mw > 500,
            logp > 5,
            hbd > 5,
            hba > 10
        ])
        
        veber_compliant = rotatable <= 10 and tpsa <= 140
        
        lead_like = (mw <= 350 and logp <= 3.5 and rotatable <= 7 and 
                    hbd <= 3 and hba <= 6 and tpsa <= 90)
        
        return {
            'Lipinski_Violations': lipinski_violations,
            'Drug_Like': lipinski_violations <= 1,
            'Veber_Compliant': veber_compliant,
            'Lead_Like': lead_like
        }
    
    def generate_fingerprints(self, fp_type: str = 'morgan', **kwargs) -> List:
        """Generate molecular fingerprints"""
        print("🔍 Generating molecular fingerprints...")
        self.fingerprints = []
        
        for mol in self.compounds:
            if fp_type == 'morgan':
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                    mol, kwargs.get('radius', 2), nBits=kwargs.get('nBits', 2048)
                )
            elif fp_type == 'rdkit':
                fp = Chem.RDKFingerprint(mol, fpSize=kwargs.get('fpSize', 2048))
            elif fp_type == 'maccs':
                fp = AllChem.GetMACCSKeysFingerprint(mol)
            else:
                raise ValueError(f"Unsupported fingerprint type: {fp_type}")
            
            self.fingerprints.append(fp)
        
        print(f"   ✅ Generated {fp_type} fingerprints for all compounds")
        return self.fingerprints
    
    def filter_drug_like(self, criteria: str = 'lipinski') -> List[int]:
        """Filter compounds based on drug-likeness"""
        print(f"🏥 Filtering compounds using {criteria} criteria...")
        
        if self.descriptors_df is None:
            self.calculate_descriptors()
        
        if criteria == 'lipinski':
            mask = self.descriptors_df['Drug_Like']
        elif criteria == 'veber':
            mask = self.descriptors_df['Veber_Compliant']
        elif criteria == 'lead_like':
            mask = self.descriptors_df['Lead_Like']
        else:
            raise ValueError(f"Unknown criteria: {criteria}")
        
        drug_like_indices = self.descriptors_df[mask].index.tolist()
        self.drug_like_compounds = [self.compounds[i] for i in drug_like_indices]
        
        print(f"   ✅ Found {len(self.drug_like_compounds)} drug-like compounds "
              f"out of {len(self.compounds)} total ({len(self.drug_like_compounds)/len(self.compounds)*100:.1f}%)")
        
        return drug_like_indices
    
    def virtual_screening(self, query_mol, similarity_threshold: float = 0.7, 
                         top_n: int = 100) -> List[Tuple[int, float]]:
        """Find molecules similar to a query molecule"""
        print("🎯 Performing virtual screening...")
        
        if not self.fingerprints:
            self.generate_fingerprints()
        
        query_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=2048)
        similarities = []
        
        for i, fp in enumerate(self.fingerprints):
            sim = DataStructs.TanimotoSimilarity(query_fp, fp)
            if sim >= similarity_threshold:
                similarities.append((i, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        query_name = query_mol.GetProp('_Name') if query_mol.HasProp('_Name') else 'Query'
        print(f"   ✅ Found {len(similarities)} compounds similar to {query_name}")
        print(f"   📊 Returning top {min(top_n, len(similarities))} matches")
        
        return similarities[:top_n]
    
    def build_qsar_model(self, activity_data: Dict[str, float], 
                        test_size: float = 0.2) -> Dict:
        """Build a prediction model (QSAR)"""
        print("🔮 Building QSAR prediction model...")
        
        if self.descriptors_df is None:
            self.calculate_descriptors()
        
        X_data = []
        y_data = []
        
        for smiles, activity in activity_data.items():
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                for idx, row in self.descriptors_df.iterrows():
                    if row['SMILES'] == smiles:
                        descriptor_cols = [col for col in self.descriptors_df.columns 
                                         if col not in ['CompoundID', 'Name', 'SMILES', 
                                                       'Lipinski_Violations', 'Drug_Like', 
                                                       'Veber_Compliant', 'Lead_Like']]
                        X_data.append(row[descriptor_cols].values)
                        y_data.append(activity)
                        break
        
        if len(X_data) < 5:
            print("   ⚠️  Not enough training data for reliable model!")
            return {}
        
        X = np.array(X_data)
        y = np.array(y_data)
        
        nan_mask = (~np.isnan(X).any(axis=1)) & (~np.isnan(y))
        X = X[nan_mask]
        y = y[nan_mask]
        
        print(f"   📊 Training on {len(X)} compounds")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        self.qsar_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.qsar_model.fit(X_train, y_train)
        
        y_pred_train = self.qsar_model.predict(X_train)
        y_pred_test = self.qsar_model.predict(X_test)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        results = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'feature_importance': dict(zip(descriptor_cols, self.qsar_model.feature_importances_))
        }
        
        print(f"   ✅ Model trained successfully!")
        print(f"   📈 Test R² Score: {test_r2:.3f}")
        print(f"   📉 Test RMSE: {test_rmse:.3f}")
        
        return results
    
    def predict_activity(self, smiles_list: List[str]) -> List[float]:
        """Predict biological activity for new compounds"""
        if self.qsar_model is None:
            raise ValueError("❌ QSAR model not trained yet! Run build_qsar_model() first.")
        
        print(f"🔮 Predicting activity for {len(smiles_list)} new compounds...")
        
        predictions = []
        descriptor_cols = [col for col in self.descriptors_df.columns 
                          if col not in ['CompoundID', 'Name', 'SMILES', 'Lipinski_Violations', 
                                        'Drug_Like', 'Veber_Compliant', 'Lead_Like']]
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                desc_values = []
                desc_functions = {
                    'MolWt': Descriptors.MolWt,
                    'LogP': Descriptors.MolLogP,
                    'NumHDonors': Descriptors.NumHDonors,
                    'NumHAcceptors': Descriptors.NumHAcceptors,
                    'TPSA': Descriptors.TPSA,
                    'NumRotatableBonds': Descriptors.NumRotatableBonds,
                    'NumAromaticRings': Descriptors.NumAromaticRings,
                    'NumAliphaticRings': Descriptors.NumAliphaticRings,
                    'NumHeteroatoms': Descriptors.NumHeteroatoms,
                    'FractionCSP3': Descriptors.FractionCSP3,
                    'HeavyAtomCount': Descriptors.HeavyAtomCount,
                    'RingCount': Descriptors.RingCount,
                    'MolMR': Descriptors.MolMR,
                    'BalabanJ': Descriptors.BalabanJ,
                    'BertzCT': Descriptors.BertzCT
                }
                
                for col in descriptor_cols:
                    if col in desc_functions:
                        try:
                            desc_values.append(desc_functions[col](mol))
                        except:
                            desc_values.append(0)
                    else:
                        desc_values.append(0)
                
                pred = self.qsar_model.predict([desc_values])[0]
                predictions.append(pred)
            else:
                predictions.append(np.nan)
        
        print(f"   ✅ Predictions completed!")
        return predictions
    
    def analyze_scaffolds(self) -> Dict:
        """Analyze molecular scaffolds"""
        print("🏗️ Analyzing molecular scaffolds...")
        scaffolds = {}
        
        for i, mol in enumerate(self.compounds):
            try:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smiles = Chem.MolToSmiles(scaffold)
                
                if scaffold_smiles not in scaffolds:
                    scaffolds[scaffold_smiles] = []
                scaffolds[scaffold_smiles].append(i)
            except:
                continue
        
        scaffold_freq = {k: len(v) for k, v in scaffolds.items()}
        sorted_scaffolds = sorted(scaffold_freq.items(), 
                                key=lambda x: x[1], reverse=True)
        
        print(f"   ✅ Found {len(scaffolds)} unique scaffolds")
        if sorted_scaffolds:
            print(f"   📊 Most common scaffold appears in {sorted_scaffolds[0][1]} compounds")
        
        return {
            'scaffolds': scaffolds,
            'frequencies': scaffold_freq,
            'top_scaffolds': sorted_scaffolds[:10]
        }
    
    def create_visualization_dashboard(self) -> None:
        """Create comprehensive visualization dashboard"""
        print("📊 Creating visualization dashboard...")
        
        if self.descriptors_df is None:
            self.calculate_descriptors()
        
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'{self.project_name} - Drug Discovery Dashboard', fontsize=16, fontweight='bold')
            
            # 1. Molecular Weight Distribution
            axes[0, 0].hist(self.descriptors_df['MolWt'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].axvline(500, color='red', linestyle='--', linewidth=2, label='Lipinski Limit (500)')
            axes[0, 0].set_xlabel('Molecular Weight (Da)')
            axes[0, 0].set_ylabel('Number of Compounds')
            axes[0, 0].set_title('Molecular Weight Distribution')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Chemical Space (LogP vs TPSA)
            drug_like_colors = self.descriptors_df['Drug_Like'].astype(int)
            scatter = axes[0, 1].scatter(self.descriptors_df['LogP'], 
                                       self.descriptors_df['TPSA'],
                                       c=drug_like_colors,
                                       cmap='RdYlGn', alpha=0.7, s=50)
            axes[0, 1].set_xlabel('LogP (Lipophilicity)')
            axes[0, 1].set_ylabel('TPSA (Polar Surface Area)')
            axes[0, 1].set_title('Chemical Space Map')
            axes[0, 1].grid(True, alpha=0.3)
            cbar = plt.colorbar(scatter, ax=axes[0, 1])
            cbar.set_label('Drug-like (0=No, 1=Yes)')
            
            # 3. Drug-likeness Distribution
            drug_like_counts = self.descriptors_df['Drug_Like'].value_counts()
            labels = []
            colors = []
            values = []
            
            if False in drug_like_counts.index:
                labels.append('Not Drug-like')
                colors.append('lightcoral')
                values.append(drug_like_counts[False])
            
            if True in drug_like_counts.index:
                labels.append('Drug-like')
                colors.append('lightgreen')
                values.append(drug_like_counts[True])
            
            if values:
                axes[0, 2].pie(values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
                axes[0, 2].set_title('Drug-likeness Distribution')
            else:
                axes[0, 2].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axes[0, 2].transAxes)
                axes[0, 2].set_title('Drug-likeness Distribution')
            
            # 4. Descriptor Correlations
            desc_cols = ['MolWt', 'LogP', 'TPSA', 'NumRotatableBonds', 'NumAromaticRings']
            corr_matrix = self.descriptors_df[desc_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, ax=axes[1, 0])
            axes[1, 0].set_title('Descriptor Correlations')
            
            # 5. Molecular Complexity vs Size
            axes[1, 1].scatter(self.descriptors_df['BertzCT'], 
                              self.descriptors_df['HeavyAtomCount'], 
                              alpha=0.6, s=50, color='purple')
            axes[1, 1].set_xlabel('Bertz Complexity Index')
            axes[1, 1].set_ylabel('Heavy Atom Count')
            axes[1, 1].set_title('Molecular Complexity vs Size')
            axes[1, 1].grid(True, alpha=0.3)
            
            # 6. Lipinski Violations
            viol_counts = self.descriptors_df['Lipinski_Violations'].value_counts().sort_index()
            bars = axes[1, 2].bar(viol_counts.index, viol_counts.values, 
                                 color='orange', alpha=0.7, edgecolor='black')
            axes[1, 2].set_xlabel('Number of Lipinski Violations')
            axes[1, 2].set_ylabel('Number of Compounds')
            axes[1, 2].set_title('Lipinski Rule Violations')
            axes[1, 2].grid(True, alpha=0.3)
            
            for bar in bars:
                height = bar.get_height()
                axes[1, 2].text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height)}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save the plot
            filename = f"{self.project_name}_dashboard.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"   ✅ Dashboard saved as: {filename}")
            plt.show()
            
        except Exception as e:
            print(f"   ⚠️ Error creating dashboard: {e}")
            print("   Dashboard creation failed, but analysis continues...")
    
    def generate_comprehensive_report(self, filename: str = None) -> None:
        """Generate detailed analysis report"""
        if filename is None:
            filename = f'{self.project_name}_report.txt'
            
        print(f"📝 Generating comprehensive report...")
        
        try:
            with open(filename, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write(f"  {self.project_name.upper()} - DRUG DISCOVERY ANALYSIS REPORT\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d at %H:%M:%S')}\n\n")
                
                # Dataset Overview
                f.write("📊 DATASET OVERVIEW\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total compounds analyzed: {len(self.compounds)}\n")
                
                if self.descriptors_df is not None:
                    drug_like = self.descriptors_df['Drug_Like'].sum()
                    veber_compliant = self.descriptors_df['Veber_Compliant'].sum()
                    lead_like = self.descriptors_df['Lead_Like'].sum()
                    
                    f.write(f"Drug-like compounds (Lipinski): {drug_like} ({drug_like/len(self.compounds)*100:.1f}%)\n")
                    f.write(f"Veber compliant compounds: {veber_compliant} ({veber_compliant/len(self.compounds)*100:.1f}%)\n")
                    f.write(f"Lead-like compounds: {lead_like} ({lead_like/len(self.compounds)*100:.1f}%)\n\n")
                    
                    # Descriptor Statistics
                    f.write("📏 MOLECULAR DESCRIPTOR STATISTICS\n")
                    f.write("-" * 40 + "\n")
                    key_descriptors = ['MolWt', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds']
                    desc_stats = self.descriptors_df[key_descriptors].describe()
                    f.write(desc_stats.to_string())
                    f.write("\n\n")
                    
                    # Drug-likeness Analysis
                    f.write("🏥 DRUG-LIKENESS ANALYSIS\n")
                    f.write("-" * 40 + "\n")
                    lipinski_violations = self.descriptors_df['Lipinski_Violations'].value_counts().sort_index()
                    f.write("Lipinski Rule Violations Distribution:\n")
                    for violations, count in lipinski_violations.items():
                        f.write(f"  {violations} violations: {count} compounds ({count/len(self.compounds)*100:.1f}%)\n")
                    f.write("\n")
                
                # Scaffold Analysis
                scaffold_analysis = self.analyze_scaffolds()
                f.write("🏗️ MOLECULAR SCAFFOLD ANALYSIS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Unique scaffolds found: {len(scaffold_analysis['scaffolds'])}\n")
                if scaffold_analysis['frequencies']:
                    f.write(f"Most diverse compounds per scaffold: {max(scaffold_analysis['frequencies'].values())}\n\n")
                    f.write("Top 10 Most Common Scaffolds:\n")
                    for i, (scaffold, count) in enumerate(scaffold_analysis['top_scaffolds'][:10], 1):
                        f.write(f"  {i:2d}. {scaffold} ({count} compounds)\n")
                f.write("\n")
                
                # Recommendations
                f.write("🎯 RECOMMENDATIONS\n")
                f.write("-" * 40 + "\n")
                f.write("1. Perform virtual screening with known active compounds\n")
                f.write("2. Build QSAR models if activity data is available\n")
                f.write("3. Analyze molecular scaffolds for series expansion\n")
                f.write("4. Consider ADMET property prediction\n")
                f.write("5. Prioritize compounds for experimental testing\n\n")
                
                f.write("=" * 80 + "\n")
                f.write("End of Report\n")
                f.write("=" * 80 + "\n")
            
            print(f"   ✅ Comprehensive report saved as: {filename}")
            
        except Exception as e:
            print(f"   ⚠️ Error generating report: {e}")
    
    def save_pipeline(self, filename: str = None) -> None:
        """Save the entire pipeline for future use"""
        if filename is None:
            filename = f'{self.project_name}_pipeline.pkl'
            
        print(f"💾 Saving pipeline...")
        
        try:
            pipeline_data = {
                'project_name': self.project_name,
                'compounds': [Chem.MolToSmiles(mol) for mol in self.compounds],
                'compound_names': [mol.GetProp('_Name') for mol in self.compounds],
                'descriptors_df': self.descriptors_df,
                'qsar_model': self.qsar_model,
            }
            
            with open(filename, 'wb') as f:
                pickle.dump(pipeline_data, f)
            
            print(f"   ✅ Pipeline saved as: {filename}")
            
        except Exception as e:
            print(f"   ⚠️ Error saving pipeline: {e}")