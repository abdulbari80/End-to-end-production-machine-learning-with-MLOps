from flask import Flask, request, render_template
from prediction import Prediction


application = Flask(__name__)
app = application


@app.route("/", methods=["GET", "POST"], endpoint="predict_user_salary")
def predict():
    if request.method == "POST":
        form_data = request.form.to_dict()
        print("Received POST data:", form_data)

        # Match all field names exactly as in HTML
        experience_level = form_data.get("experience_level", "").strip()
        employment_type = form_data.get("employment_type", "").strip()
        remote_ratio = form_data.get("remote_ratio", "").strip()
        company_size = form_data.get("company_size", "").strip()
        job_title_freq = form_data.get("job_title_freq", "").strip()
        employee_residence_top = form_data.get("employee_residence_top", "").strip()
        company_location_top = form_data.get("company_location_top", "").strip()

        # Validation
        missing_fields = []
        if not experience_level: missing_fields.append("Experience Level")
        if not employment_type: missing_fields.append("Employment Type")
        if not remote_ratio: missing_fields.append("Office Attendance")
        if not company_size: missing_fields.append("Company Size")
        if not job_title_freq: missing_fields.append("Job Role")
        if not employee_residence_top: missing_fields.append("Your Residence")
        if not company_location_top: missing_fields.append("Company Location")

        if missing_fields:
            print(f"Missing fields: {missing_fields}")
            return render_template(
                "index.html",
                results="Please fill in all fields before asking Maban.",
            )

        
        # Prediction
        try:
            predictor = Prediction(
                experience_level=experience_level,
                employment_type=employment_type,
                remote_ratio=remote_ratio,
                company_size=company_size,
                job_title_freq=job_title_freq,
                employee_residence_top=employee_residence_top,
                company_location_top=company_location_top,
            )
            result = predictor.get_prediction()
            print("Model prediction:", result)
        except Exception as e:
            print("Prediction error:", e)
            return render_template(
                "index.html",
                results=f"Error during prediction: {e}",
            )

        # Format result nicely
        try:
            result_value = float(result)
        except ValueError:
            result_value = 0
        # warn for unusual input combination when too low salary prediction
        if result_value < 6021:
            message = "Hmm... that seems too low. Please check your input combination."
        else:
            salary_k = int(round(result_value / 1000, 0))
            message = (
                f"<strong style='font-size:1.5em; color:#7B61FF;'>{salary_k}K</strong> US$ a year, mate!"
            )

        return render_template(
            "index.html",
            results=message,
        )

    return render_template("index.html", results=None)

 #--- Utility / health routes ---
@app.route("/health")
def health():
    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
