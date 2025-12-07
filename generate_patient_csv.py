"""
Generate standalone patient database CSV file
This is an independent file, not connected to the clinical trials system
"""
import csv
import random
from datetime import datetime, timedelta

# Common diseases pool
diseases_pool = [
    "Type 2 Diabetes",
    "Hypertension",
    "Hyperlipidemia",
    "Coronary Artery Disease",
    "Chronic Kidney Disease",
    "Heart Failure",
    "Asthma",
    "COPD",
    "Depression",
    "Anxiety Disorder",
    "Rheumatoid Arthritis",
    "Osteoarthritis",
    "Obesity",
    "Atrial Fibrillation",
    "Stroke",
    "Cancer (Remission)",
    "Hypothyroidism",
    "Osteoporosis",
    "Migraine",
    "Chronic Back Pain"
]

def generate_patient_data(num_patients=500):
    """Generate synthetic patient data"""
    patients = []
    
    for i in range(num_patients):
        subject_id = f"PT{10000 + i}"
        gender = random.choice(["M", "F"])
        
        # Age distribution - realistic for clinical trials
        age = random.randint(25, 85)
        
        # Weight based on gender (realistic distribution)
        if gender == "M":
            weight = round(random.uniform(55, 130), 1)  # kg
        else:
            weight = round(random.uniform(45, 110), 1)  # kg
        
        # Height based on gender (realistic distribution)
        if gender == "M":
            height = round(random.uniform(160, 195), 1)  # cm
        else:
            height = round(random.uniform(150, 180), 1)  # cm
        
        # Assign 1-3 diseases randomly
        num_diseases = random.randint(1, 3)
        patient_diseases = random.sample(diseases_pool, num_diseases)
        diseases_str = "; ".join(patient_diseases)
        
        # BMI calculation
        bmi = round(weight / ((height/100) ** 2), 1)
        
        # Last visit date (within last 2 years)
        days_ago = random.randint(1, 730)
        last_visit = (datetime.now() - timedelta(days=days_ago))
        # Ensure 2024 or earlier
        if last_visit.year >= 2025:
            last_visit = last_visit.replace(year=2024)
        last_visit_str = last_visit.strftime("%Y-%m-%d")
        
        patients.append({
            "Subject_ID": subject_id,
            "Gender": gender,
            "Age": age,
            "Weight_kg": weight,
            "Height_cm": height,
            "BMI": bmi,
            "Diseases": diseases_str,
            "Last_Visit": last_visit_str
        })
    
    return patients

def save_to_csv(patients, filename="patient_database.csv"):
    """Save patient data to CSV file"""
    fieldnames = ["Subject_ID", "Gender", "Age", "Weight_kg", "Height_cm", "BMI", "Diseases", "Last_Visit"]
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(patients)
    
    print(f"✓ Created {filename} with {len(patients)} patient records")
    print(f"✓ Columns: {', '.join(fieldnames)}")
    print(f"✓ File is ready to use and independent of the clinical trials system")

if __name__ == "__main__":
    print("=" * 60)
    print("PATIENT DATABASE GENERATOR")
    print("=" * 60)
    print()
    
    # Generate patient data
    num_patients = 500
    print(f"Generating {num_patients} synthetic patient records...")
    patients = generate_patient_data(num_patients)
    
    # Save to CSV
    save_to_csv(patients, "patient_database.csv")
    
    # Show sample
    print()
    print("Sample records:")
    print("-" * 60)
    for i, patient in enumerate(patients[:5], 1):
        print(f"{i}. {patient['Subject_ID']} | {patient['Gender']} | Age: {patient['Age']} | "
              f"Weight: {patient['Weight_kg']}kg | Height: {patient['Height_cm']}cm | "
              f"BMI: {patient['BMI']} | Diseases: {patient['Diseases'][:50]}...")
    
    print()
    print("=" * 60)
    print("GENERATION COMPLETE!")
    print("=" * 60)
