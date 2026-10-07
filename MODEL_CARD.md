# SarcopeniaAI Model Card

## Purpose

SarcopeniaAI is a research-oriented multimodal deep-learning
prototype for estimating low-muscle-strength risk.

It combines ultrasound information from:

- Rectus Femoris (RF)
- Vastus Lateralis (VL)
- Tibialis Anterior (TA)

with four clinical variables:

- Age
- Height
- Weight
- Sex

## Architecture

Image branch:
ResNet18 ultrasound encoder.

Clinical branch:
Multilayer perceptron.

Fusion:
Image and clinical representations are combined using
feature-level fusion.

## Target

Binary low-muscle-strength research target.

## Explainability

Grad-CAM is used to visualize ultrasound regions that
influence model predictions.

## Important Limitation

The development dataset contains a very small number of
patients. The final multimodal test subset contains only
seven complete RF/VL/TA patients.

The model showed poor generalization and should NOT be
interpreted as a validated clinical diagnostic system.

## Intended Use

Educational, portfolio, and research demonstration only.

NOT FOR CLINICAL DIAGNOSIS.
