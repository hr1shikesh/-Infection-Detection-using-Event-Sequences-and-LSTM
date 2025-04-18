import random

def generate_sepsis_dataset(n=20):
    """Generates a dataset with timestamps for sepsis patients."""
    dataset = []
    time = 0  # Start time in minutes
    for _ in range(n):
        data = {
            "Time": time,
            "HR Variability": random.randint(18, 25),  # Increased in Sepsis
            "BP": random.randint(120, 140),  # Slight drop possible
            "Body Temp": round(random.uniform(38.0, 40.0), 1),  # High Fever
            "SpO2": random.randint(92, 97)  # Possible fluctuation
        }
        dataset.append(data)
        time += 10  # Logging every 10 minutes
    return dataset

def detect_events(patient_records):
    """Detects events based on rate thresholds and generates event chains."""
    event_chain = []
    event_durations = {}
    previous_events = set()
    
    # Thresholds for significant change per 10 mins
    HRV_THRESHOLD = 2  # Change >= 2 in 10 mins
    BP_THRESHOLD = 5  # Drop >= 5 mmHg in 10 mins
    TEMP_THRESHOLD = 0.5  # Change >= 0.5°C in 10 mins
    SPO2_THRESHOLD = 2  # Drop >= 2% in 10 mins
    LONG_DURATION = 30  # Threshold for Long classification (minutes)
    
    for i in range(1, len(patient_records)):
        prev_record = patient_records[i - 1]
        curr_record = patient_records[i]
        temp_chain = []
        time_diff = curr_record["Time"] - prev_record["Time"]
        
        # Ensure time_diff is exactly 10 minutes before using thresholds
        if time_diff != 10:
            continue
        
        current_events = set()
        
        # HR Variability significant change
        hrv_change = curr_record["HR Variability"] - prev_record["HR Variability"]
        if abs(hrv_change) >= HRV_THRESHOLD:
            event = "High HR Variability" if hrv_change > 0 else "Low HR Variability"
            current_events.add(event)
        
        # BP significant drop
        bp_change = curr_record["BP"] - prev_record["BP"]
        if bp_change <= -BP_THRESHOLD:
            current_events.add("Low BP")
        elif bp_change >= BP_THRESHOLD:
            current_events.add("High BP")
        
        # Body Temp significant change
        temp_change = curr_record["Body Temp"] - prev_record["Body Temp"]
        if abs(temp_change) >= TEMP_THRESHOLD:
            if temp_change > 0 and curr_record["Body Temp"] >= 39.0:
                current_events.add("High Body Temp")
            elif temp_change < 0 and prev_record["Body Temp"] >= 39.0:
                current_events.add("Low Body Temp")
        
        # SpO2 significant drop
        spo2_change = curr_record["SpO2"] - prev_record["SpO2"]
        if spo2_change <= -SPO2_THRESHOLD:
            current_events.add("Low SpO2")
        elif spo2_change >= SPO2_THRESHOLD:
            current_events.add("High SpO2")
        
        # Track duration of events
        for event in current_events:
            if event in event_durations:
                event_durations[event] += 10
            else:
                event_durations[event] = 10
        
        # Classify events as Short or Long
        formatted_events = []
        for event in current_events:
            duration = event_durations[event]
            label = "Long" if duration >= LONG_DURATION else "Short"
            formatted_events.append(f"{event} ({label})")
        
        # Remove negated events (e.g., High Fever followed by Low Fever)
        for prev_event in previous_events:
            if prev_event.startswith("High") and prev_event.replace("High", "Low") in current_events:
                current_events.discard(prev_event)
            elif prev_event.startswith("Low") and prev_event.replace("Low", "High") in current_events:
                current_events.discard(prev_event)
        
        if formatted_events:
            event_chain.append(" + ".join(formatted_events))
        previous_events = current_events
        
    return " → ".join(event_chain) if event_chain else "No significant change"

# Generate dataset and test
dataset = generate_sepsis_dataset()
for record in dataset:
    print(record)

# Process event detection
final_event_chain = detect_events(dataset)
print("\nFinal Event Chain for Patient:")
print(final_event_chain)