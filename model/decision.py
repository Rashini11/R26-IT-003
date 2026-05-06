def get_recommendation(prediction):
    if prediction == 0:
        return "Corrosion detected → Repainting required"

    elif prediction == 1:
        return "Crack detected → Structural repair required"

    elif prediction == 2:
        return "Biofouling detected → Cleaning required"