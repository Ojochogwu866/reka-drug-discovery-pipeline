# Drug Discovery Pipeline

A computational drug discovery pipeline built with Python and RDKit for molecular analysis and screening.

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org)
[![RDKit](https://img.shields.io/badge/RDKit-2023+-green.svg)](https://rdkit.org)

## Features

- Load and analyze molecules from SMILES
- Calculate molecular properties and drug-likeness
- Molecular fingerprinting and similarity search
- QSAR modeling for activity prediction
- Generate comprehensive reports

## Quick Start

```bash
# Clone repository
git clone https://github.com/Ojochogwu866/reka-drug-discovery-pipeline.git
cd reka-drug-discovery-pipeline

# Install dependencies
conda install -c conda-forge rdkit pandas numpy matplotlib seaborn scikit-learn

# Run demo
python demo.py
```

## Usage

```python
from drug_discovery_pipeline import DrugDiscoveryPipeline

# Create pipeline
pipeline = DrugDiscoveryPipeline("My_Project")

# Load molecules
molecules = ['CCO', 'CC(=O)O', 'c1ccccc1']
pipeline.load_compounds(molecules)

# Run analysis
pipeline.calculate_descriptors()
pipeline.filter_drug_like('lipinski')
pipeline.generate_fingerprints()

# Generate report
pipeline.generate_comprehensive_report()
```

## Project Structure

```
├── drug_discovery_pipeline.py    # Main pipeline
├── demo.py                       # Full demo with visualizations
├── simple_demo.py               # Basic demo (error-safe)
├── requirements.txt             # Dependencies
└── README.md
```

## Drug-Likeness Rules

Uses Lipinski's Rule of Five:
- Molecular weight ≤ 500 Da
- LogP ≤ 5
- Hydrogen donors ≤ 5
- Hydrogen acceptors ≤ 10

## Sample Output

```
📦 Loading compounds... ✅ Successfully loaded 9 compounds
📏 Calculating descriptors... ✅ Calculated 15 descriptors
🏥 Filtering drug-like... ✅ Found 9/9 drug-like compounds (100%)
🔍 Generating fingerprints... ✅ Generated morgan fingerprints
🎯 Virtual screening... ✅ Found similar compounds
🔮 Building QSAR model... ✅ Model trained (R² = 0.847)
```

## Troubleshooting

**Import errors:** Install RDKit with `conda install -c conda-forge rdkit`

**Visualization issues:** Use `simple_demo.py` instead of `demo.py`

**QSAR errors:** Need at least 5 molecules with activity data

## License

MIT License - Educational use encouraged

---

*⭐ Star if this helped you learn computational drug discovery!*
