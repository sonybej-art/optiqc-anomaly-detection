---
title: OptiQC Anomaly Detection
emoji: 🔍
colorFrom: teal
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# OptiQC — Edge Anomaly Detection Demo

Live demo of the OptiQC industrial visual-inspection API. Upload a bottle
image and see the anomaly score, PASS/FAIL decision, and defect heatmap in
real time.

- **Demo UI:** `/demo`
- **API docs (Swagger):** `/docs`
- **Health check:** `GET /`
- **Inspection endpoint:** `POST /api/v1/inspect`

Full write-up, source, and the accompanying research report:
https://github.com/javito350/Object_Detector_For_Control_Quality_for_Factories
