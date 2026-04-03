# Health, Biometric & Sentiment Adaptation (Empathetic Companion)

## Overview
A true companion understands how you are feeling and adapts its behavior accordingly. By integrating with biometric wearables (Apple Watch, Oura Ring, Whoop) and analyzing your text/voice sentiment, the AI adjusts its tone, verbosity, and proactiveness to support your current state.

## Key Capabilities
- **Biometric Integration**: Connects to health APIs to monitor sleep quality, resting heart rate, and stress levels (HRV).
- **Tone Adaptation**: If your biometrics indicate high stress or poor sleep, the AI becomes more concise, speaks in a softer tone, and avoids bringing up non-urgent tasks.
- **Proactive Wellness**: "I noticed your HRV dropped and you only slept 4 hours. I've automatically pushed your non-essential morning meetings to the afternoon. Take it easy."
- **Sentiment Analysis**: Detects frustration in your text or voice. If you are struggling with a coding bug and getting angry, it shifts from "giving instructions" to "calmly debugging with you step-by-step."

## Implementation Plan

### Phase 1: Sentiment & Behavioral Analysis
1. **Real-time Sentiment Tool**: Add a lightweight sentiment analysis pass on user inputs (detecting frustration, joy, fatigue, urgency).
2. **Dynamic System Prompts**: Modify the agent's core system prompt dynamically based on the detected sentiment (e.g., append "The user is frustrated; be extremely direct, helpful, and reassuring").

### Phase 2: Wearable API Integration
1. **Data Ingestion**: Integrate with Google Fit, Apple HealthKit (via a companion app), or the Oura API to securely sync daily biometric summaries.
2. **State Machine**: Create a "User State" profile (e.g., State: Fatigued, State: Highly Focused) that the agent checks before initiating any proactive conversation.

### Phase 3: Automated Intervention
1. **Schedule Protection**: Give the AI the ability to interact with the scheduling feature (if implemented) to block off "recovery time" if burnout indicators are high.
2. **Companion Check-ins**: "You've been coding for 6 hours straight and your heart rate is elevated. Want me to order some food, or should we take a 10-minute break?"