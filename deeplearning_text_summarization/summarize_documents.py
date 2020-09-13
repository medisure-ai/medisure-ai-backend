from summarizer import Summarizer
body = """Aetna 
P.O. Box 3013 Mad Stop U12W Blue Bell, PA 19422-0763
Dear Mr. Powell
Thank you for applying for an Aetna Advantage Plans for Individuals, Families, and the Self-Employed. After carefully evaluating your application, we are unable to offer coverage to you at this time.
Coverage is not guaranteed under an Aetna Advantage Plan for Individuals, Families and the Self-Employed All applicants and/or their dependents are medically underwritten to determine each individual's risk category. We offer various risk categories based on each applicant's medical risk factors. After careful review of your application, we are unable to offer coverage and have declined the following applicant for the following reason(s):
Based on the medical information provided, your septoplasty was deemed cosmetic and not medically necessary.

This determination is based on review of your medical conditions and associated treatment, which may include medications. This information may have come from your application, phone interview or medical records.
You may want to apply again in the future. If you do, we will review your medical history. Medical factors that we did not review this time may be considered. This means you might be offered a higher rate or be declined again.

You have the right to appeal our decision You may submit a request for an appeal in writing. Your appeal must be received within 180 days of the underwriting decision."
"""
model = Summarizer()
#result1 = model(body, ratio=0.2)  # Specified with ratio
result2 = model(body, num_sentences=3)  # Will return 3 sentences 

#print(result1)
print(result2)