from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import matplotlib
matplotlib.use('Agg') # Set non-interactive backend
import matplotlib.pyplot as plt
import os
import time

class ReportGenerator:
    PRECAUTIONS = {
        "COVID-19": [
            "Isolate immediately to prevent spread.",
            "Monitor oxygen levels regularly.",
            "Consult a doctor for antiviral medication.",
            "Rest and stay hydrated."
        ],
        "PNEUMONIA": [
            "Complete the full course of antibiotics if prescribed.",
            "Drink plenty of fluids to loosen secretions.",
            "Use a humidifier to keep air moist.",
            "Rest and avoid strenuous activity."
        ],
        "TUBERCULOSIS": [
            "Strictly adhere to the long-term medication regimen.",
            "Ensure good ventilation in living spaces.",
            "Cover mouth while coughing to protect others.",
            "Eat a nutritious diet to boost immunity."
        ],
        "NORMAL": [
            "Maintain a healthy lifestyle.",
            "Avoid smoking and pollutants.",
            "Exercise regularly to improve lung capacity.",
            "Go for regular health check-ups."
        ]
    }

    @staticmethod
    def generate_graph(probabilities, output_path):
        """
        Generates a bar graph of disease probabilities.
        probabilities: dict like {'COVID-19': 0.1, 'NORMAL': 0.8, ...}
        """
        try:
            diseases = list(probabilities.keys())
            probs = list(probabilities.values())

            plt.figure(figsize=(6, 4))
            bars = plt.bar(diseases, probs, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
            plt.xlabel('Disease')
            plt.ylabel('Confidence Score')
            plt.title('Disease Probability Distribution')
            plt.ylim(0, 1)
            
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.2f}", va='bottom')

            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            return output_path
        except Exception as e:
            print(f"Error generating graph: {e}")
            return None

    @staticmethod
    def determine_stage(prediction, confidence):
        if prediction == "NORMAL":
            return "Healthy", colors.green
        
        if confidence < 0.70:
            return "Model Certainty: Low", colors.orange
        elif confidence < 0.85:
            return "Model Certainty: Moderate", colors.orange
        elif confidence < 0.95:
            return "Model Certainty: High", colors.red
        else:
            return "Model Certainty: Very High", colors.red

    @staticmethod
    def generate_pdf(patient_data, prediction, confidence, probabilities, image_path, graph_path, output_filename, twin=None, history=None):
        try:
            c = canvas.Canvas(output_filename, pagesize=letter)
            width, height = letter

            # ================= PAGE 1: DIAGNOSIS =================
            # --- Logo ---
            logo_path = "resources/logo.png"
            if os.path.exists(logo_path):
                try:
                    logo_width = 150
                    logo_height = 50
                    c.drawImage(logo_path, (width - logo_width) / 2, height - 80, width=logo_width, height=logo_height, preserveAspectRatio=True, mask='auto')
                except Exception as e:
                    print(f"Error drawing logo: {e}")

            # --- Header ---
            c.setFillColor(colors.darkblue)
            c.setFont("Helvetica-Bold", 20)
            c.drawCentredString(width / 2, height - 110, "Lung Disease Classification Report")
            
            c.setFillColor(colors.black)
            c.setFont("Helvetica", 10)
            c.drawCentredString(width / 2, height - 130, f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            c.line(50, height - 140, width - 50, height - 140)

            # --- Patient Details ---
            y_pos = height - 180
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, y_pos, "Patient Details")
            y_pos -= 25
            
            c.setFont("Helvetica", 12)
            c.drawString(50, y_pos, f"Patient ID: {patient_data.get('id', 'N/A')}")
            c.drawString(300, y_pos, f"Name: {patient_data.get('name', 'N/A')}")
            y_pos -= 20
            c.drawString(50, y_pos, f"Age: {patient_data.get('age', 'N/A')}")
            c.drawString(300, y_pos, f"History: {patient_data.get('history', 'N/A')}")
            
            # --- Diagnosis ---
            y_pos -= 50
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, y_pos, "Diagnosis Result")
            y_pos -= 30
            
            # Get Certainty Stage and Color
            stage, color = ReportGenerator.determine_stage(prediction, confidence)
            
            c.setFillColor(color)
            c.setFont("Helvetica-Bold", 18)
            c.drawString(50, y_pos, f"{prediction} ({confidence:.1%} confidence)")
            
            y_pos -= 25
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, y_pos, f"Evaluation: {stage}")
            
            c.setFillColor(colors.black)

            # --- Images ---
            y_pos -= 240
            # X-Ray Image
            if os.path.exists(image_path):
                c.drawImage(image_path, 50, y_pos, width=250, height=200, preserveAspectRatio=True)
                c.setFont("Helvetica-Oblique", 10)
                c.drawString(50, y_pos - 15, "Enhanced X-Ray Scan")

            # Probability Graph
            if graph_path and os.path.exists(graph_path):
                c.drawImage(graph_path, 320, y_pos, width=250, height=200, preserveAspectRatio=True)
                c.drawString(320, y_pos - 15, "Probability Analysis")

            # --- Precautions ---
            y_pos -= 45
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, y_pos, "Recommended Precautions")
            y_pos -= 22
            c.setFont("Helvetica", 11)
            
            precautions = ReportGenerator.PRECAUTIONS.get(prediction, ["Consult a medical specialist."])
            for i, p in enumerate(precautions, 1):
                c.drawString(50, y_pos, f"{i}. {p}")
                y_pos -= 18

            # --- Footer ---
            c.setFont("Helvetica-Oblique", 8)
            c.drawString(50, 40, "* This report is generated by AI and must be verified by a qualified medical professional.")

            # ================= PAGE 2: AI VISUAL ANALYSIS (HEALTHY TWIN) =================
            if twin and isinstance(twin, dict):
                c.showPage()
                c.setFillColor(colors.darkblue)
                c.setFont("Helvetica-Bold", 18)
                c.drawCentredString(width / 2, height - 50, "AI Visual Analysis & Healthy Twin")
                c.line(50, height - 60, width - 50, height - 60)

                # Draw Triptych (Original | Healthy Twin | Difference Heatmap)
                triptych_path = twin.get("images", {}).get("triptych")
                y_twin = height - 290
                if triptych_path and os.path.exists(triptych_path):
                    c.drawImage(triptych_path, 50, y_twin, width=width - 100, height=210, preserveAspectRatio=True)

                # Abnormality Metrics
                y_twin -= 35
                abnormality_index = twin.get("abnormality_index", 0)
                status_message = twin.get("message", "Visual reconstruction completed.")
                status_code = twin.get("status", "unknown")

                c.setFillColor(colors.black)
                c.setFont("Helvetica-Bold", 13)
                c.drawString(50, y_twin, f"Abnormality Index: {abnormality_index} / 100")
                
                # Draw index bar
                y_twin -= 18
                c.setStrokeColor(colors.gray)
                c.setFillColor(colors.lightgrey)
                c.rect(50, y_twin, 300, 12, fill=1, stroke=1)
                bar_fill_color = colors.green if abnormality_index < 50 else (colors.orange if abnormality_index < 80 else colors.red)
                c.setFillColor(bar_fill_color)
                fill_width = min(300, max(0, int(300 * (abnormality_index / 100.0))))
                c.rect(50, y_twin, fill_width, 12, fill=1, stroke=0)

                # 95th percentile marker
                c.setStrokeColor(colors.black)
                c.line(50 + int(300 * 0.95), y_twin - 3, 50 + int(300 * 0.95), y_twin + 15)
                c.setFont("Helvetica", 8)
                c.setFillColor(colors.darkgray)
                c.drawString(50 + int(300 * 0.95) - 20, y_twin - 12, "95% Healthy Cutoff")

                # Fusion Message
                y_twin -= 35
                msg_color = colors.green if status_code == "consistent_normal" else (colors.red if status_code in ("consistent_disease", "unknown_abnormal") else colors.orange)
                c.setFillColor(msg_color)
                c.setFont("Helvetica-Bold", 12)
                c.drawString(50, y_twin, f"Clinical Fusion: {status_message}")

                # Explanation Box
                y_twin -= 65
                c.setFillColor(colors.whitesmoke)
                c.rect(50, y_twin, width - 100, 48, fill=1, stroke=1)
                c.setFillColor(colors.darkslategray)
                c.setFont("Helvetica-Bold", 9)
                c.drawString(60, y_twin + 32, "Notice on Healthy Twin Reconstruction:")
                c.setFont("Helvetica", 8.5)
                c.drawString(60, y_twin + 18, "The Healthy Twin is an AI-generated autoencoder model of normal chest anatomy.")
                c.drawString(60, y_twin + 6, "Regions with elevated error indicate deviations from healthy baselines; not a ground-truth diagnosis.")

                c.setFont("Helvetica-Oblique", 8)
                c.setFillColor(colors.black)
                c.drawString(50, 40, "* AI-generated reference, not a diagnosis.")

            # ================= PAGE 3: SCAN HISTORY (UNIVERSAL SCAN MEMORY) =================
            if history and isinstance(history, list) and len(history) > 0:
                c.showPage()
                c.setFillColor(colors.darkblue)
                c.setFont("Helvetica-Bold", 18)
                c.drawCentredString(width / 2, height - 50, "Patient Longitudinal Scan History")
                c.line(50, height - 60, width - 50, height - 60)

                y_hist = height - 90
                c.setFont("Helvetica-Bold", 10)
                c.setFillColor(colors.black)
                c.drawString(50, y_hist, "Date")
                c.drawString(140, y_hist, "Medical Center")
                c.drawString(280, y_hist, "Diagnosis")
                c.drawString(410, y_hist, "Abnormality")
                c.drawString(500, y_hist, "Trend")
                y_hist -= 5
                c.line(50, y_hist, width - 50, y_hist)

                c.setFont("Helvetica", 9)
                for item in history[:10]: # Up to 10 historical scans
                    y_hist -= 20
                    c.drawString(50, y_hist, str(item.get("date", "N/A")))
                    c.drawString(140, y_hist, str(item.get("center", "N/A"))[:22])
                    c.drawString(280, y_hist, f"{item.get('label', 'N/A')} ({item.get('confidence', 0):.0%})")
                    c.drawString(410, y_hist, f"{item.get('abnormality_index', 'N/A')}/100")
                    c.drawString(500, y_hist, str(item.get("trend", "-")))

                c.setFont("Helvetica-Oblique", 8)
                c.setFillColor(colors.black)
                c.drawString(50, 40, "* Universal Scan Memory - AI comparative records.")

            c.save()
            return output_filename
        except Exception as e:
            print(f"Error generating PDF: {e}")
            return None
