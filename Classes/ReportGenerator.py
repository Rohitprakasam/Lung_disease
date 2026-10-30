from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import matplotlib
matplotlib.use('Agg') # Set non-interactive backend
import matplotlib.pyplot as plt
import os
import time

class ReportGenerator:
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

            # --- Images (Enhanced X-Ray & Probability Analysis) ---
            y_pos -= 250
            if os.path.exists(image_path):
                c.drawImage(image_path, 50, y_pos, width=250, height=200, preserveAspectRatio=True)
                c.setFont("Helvetica-Oblique", 10)
                c.drawString(50, y_pos - 15, "Enhanced X-Ray Scan")

            if graph_path and os.path.exists(graph_path):
                c.drawImage(graph_path, 320, y_pos, width=250, height=200, preserveAspectRatio=True)
                c.drawString(320, y_pos - 15, "Probability Analysis")

            # --- Footer ---
            c.setFont("Helvetica-Oblique", 8)
            c.drawString(50, 40, "* This report is generated by AI and must be verified by a qualified medical professional.")

            # ================= SCAN HISTORY (UNIVERSAL SCAN MEMORY) =================
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
