from flask import Flask, request, jsonify
import os
from PIL import Image
import langchain
import langchain_community
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json
from rapidocr_onnxruntime import RapidOCR

app = Flask(__name__)

def match_keys_minitable(extracted_values):
    template = """
    Extract the following details from the provided OCR text:

    totalDiscount: Any discounts applied to the total amount. This is often labeled as "Discount" or "Total Discount". If no discount is present, return an empty string.
    totalGrossValue: The subtotal or gross amount before any additional charges like VAT or shipping. This is often labeled as "Sub Total", "Gross Total", or similar.
    totalAmount: The final total amount, often labeled as or near "Total", "Total Amount", or similar. This is the final figure that includes all charges.
    totalNetValue: The net amount after applying any discounts but before adding VAT and shipping. This is often labeled as "Net Total" or "Net Amount". If no discounts are applied, it might be the same as the gross value.
    totalQuantity: Total quantity is usually not directly listed on the invoice; return an empty string for this field.
    totalShippingCharge: The cost of shipping or freight. It might be labeled as "Shipping", "Shipping Charges", or "Freight".
    totalVatAmount: The VAT or tax amount. This is often labeled as "VAT", "Sales Tax", or similar.
    invoiceId: The unique ID of the invoice, often labeled as "Invoice #" or "Invoice No".
    dateOfIssue: The date when the invoice was issued, often labeled as "Date" or "Date of Issue".
    billingTo: The name and address of the customer being billed, often labeled as "Bill To".
    billingFrom: The name and address of the company issuing the invoice, often labeled as "From" or the company name.
    units: A list of all units/services provided. Each unit should include:
        - unitName: The name or description of the service or product.
        - unitPrice: The price per unit.
        - unitQuantity: The quantity of the unit/service.

    Provide the extracted values as structured JSON object.

    ### Table OCR Text:
    {extracted_values}
        
    Provide the extracted values in the following JSON format:
    
        "invoiceId": "",
        "dateOfIssue": "",
        "billingTo": "",
        "billingFrom": "",
        "units":"",
        "totalQuantity": "",
        "totalGrossValue": "",
        "totalDiscount": "",
        "totalNetValue": "",
        "totalVatAmount": "",
        "totalAmount": "",
        "totalShippingCharge": ""
        
    Output should be in JSON object format without extra characters or explanations.
    """  
    prompt_template = PromptTemplate(input_variables=["extracted_values"], template=template)
    llm = Ollama(base_url="http://127.0.0.1:11434/", model="mistral", temperature=0)

    llm_chain = LLMChain(llm=llm, prompt=prompt_template)

    result = llm_chain.generate([{"extracted_values": extracted_values}])
    llm_output = result.dict().get("generations", [])[0][0].get("text", "").strip()

    final_result = json.loads(llm_output)

    return final_result

def preprocess_image(filepath):
    
    image = Image.open(filepath)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    processed_filepath = f"{os.path.splitext(filepath)[0]}_processed.png"
    image.save(processed_filepath)
    return processed_filepath
    
def ocr(filepath):
    engine = RapidOCR()
    result, elapse = engine(filepath)
    text = '\n'.join([item[1] for item in result])
    print("Extracted Text:", text)
    print("Elapsed Time:", elapse)
    return text

@app.route('/hello', methods=['POST'])
def index():
    if 'invoice_image' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['invoice_image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    filepath = os.path.join("/tmp", file.filename)
    file.save(filepath)

    # Preprocess the image if it's a PNG
    if file.filename.lower().endswith('.png'):
        filepath = preprocess_image(filepath)

    extracted_text = ocr(filepath)
    processed_data = match_keys_minitable(extracted_text)

    os.remove(filepath)
    
    return jsonify(processed_data)

if __name__ == "__main__":
    app.run(debug=True)
