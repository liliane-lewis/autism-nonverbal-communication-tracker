# Database Schema — Non-Verbal Communication Tracker for Children with ASD

**Diagram (dbdiagram.io):**  
https://dbdiagram.io/d/autism-nonverbal-communication-tracker-68ed2f2b2e68d21b413a7a1f

This database schema belongs to the **Non-Verbal Communication Tracker for Children with ASD**, a data-driven tool designed to help therapists and parents monitor the progress of non-verbal autistic children.

## Purpose
The schema supports structured recording and analysis of behaviors such as **eye contact, gestures, vocalizations, and imitation**. It also tracks **therapy goals**, **observer roles**, **AI-generated alerts**, and **translations** for multilingual dashboards.

## Technical Overview
- PostgreSQL-based relational model  
- Fully anonymized (no personal identifiers)  
- Timezone-aware timestamps (**TIMESTAMPTZ**)  
- Optimized for **Metabase** integration and **AI-driven** analytics

## Main Entities
- **child** → anonymized child info (e.g., `code`, `diagnosis_level`)  
- **observer** → who recorded the behavior (therapist, parent, teacher)  
- **behavior_type / behavior_category** → behavior taxonomy lookups  
- **behavior_entry** → each observed event (timestamp, intensity, duration)  
- **goal / goal_event** → therapeutic objectives and status over time  
- **alert** → AI notifications (e.g., stagnation or regression detection)  
- **translation** → multilingual terms for the dashboard interface

## Machine Learning Support
The structure enables **pattern recognition**, **regression detection**, and **personalized insights** by analyzing temporal trends in `behavior_entry` and `goal_event`.

## Visualization
Aggregated views—`v_behavior_daily`, `v_behavior_summary`, and `v_goal_status`—are designed for direct connection with **Metabase** (or Power BI), simplifying progress tracking and early-intervention insights.