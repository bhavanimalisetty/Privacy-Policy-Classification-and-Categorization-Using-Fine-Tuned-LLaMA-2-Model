from django.shortcuts import render
from django import forms
from . import forms
from django.http import HttpResponse,JsonResponse
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline, AutoModelForCausalLM
from huggingface_hub import HfFolder
import torch
import nltk
import re 
import os 
import torch.nn.functional as F  # For softmax
from django.views.decorators.csrf import csrf_exempt
import json
import  time
import pandas as pd
nltk.download('punkt')

def ChatbotInputForm_View(request):
    ## This code is for printi8ng the excel file and checking the performance of the model
    print(os.getcwd())
    # file_path = "../../../Datasets/Privacy_Policy_data.xlsx"
    # results_path = "../../results/metric_quantized_fine_tunedmodel.xlsx"
    # results_path = "../../results/metric_base_line_llama2_model.xlsx"
    # results_path = "../../results/metric_fine_tuned_llama2_model.xlsx"

    # df = pd.read_excel(file_path)

    # if df.shape[1] < 2:
    #     print("❌ The file doesn't have a second column.")
    #     return

    # second_col_name = df.columns[1]
    # texts = df[second_col_name].dropna().tolist()

    # results = []

    # for idx, text in enumerate(texts):
    #     print(f"⚙️ Processing policy {idx + 1}/{len(texts)}")

    #     try:
    #         start = time.time()
    #         # output = llm_response(text, model_name="Fine Tuned Quantized Model")

    #         output = llm_response(text, model_name="Fine Tuned Model")
    #         end = time.time()

    #         elapsed = end - start
    #         tokens = len(output.split())
    #         tokens_per_sec = tokens / elapsed if elapsed > 0 else 0

    #         results.append({
    #             "Index": idx + 1,
    #             "Time (s)": round(elapsed, 2),
    #             "Tokens": tokens,
    #             "Tokens/sec": round(tokens_per_sec, 2)
    #         })

    #     except Exception as e:
    #         print(f"❌ Error on row {idx + 1}: {e}")
    #         results.append({
    #             "Index": idx + 1,
    #             "Time (s)": None,
    #             "Tokens": None,
    #             "Tokens/sec": None,
    #             "Error": str(e)
    #         })
    #     print()
    # os.makedirs("../../results", exist_ok=True)
    # results_df = pd.DataFrame(results)
    # results_df.to_excel(results_path, index=False)
    # print(f"✅ Benchmark completed. Results saved to: {results_path}")


    if request.method=="POST":
        text = request.POST.get('text')
        print("request.POST",request.POST)

        return render(request,"chatbot_form.html",)    
    return render(request, "chatbot_form.html")

def display_message():
    return HttpResponse("Hello World")

def llm_response(pp_text,model_name):
    # return "Sentence: We at Meta want you to understand what information we collect, and how we use and share it. Category: First Party Collection/Use Sentence: That’s why we encourage you to read our Privacy Policy. Category: First Party Collection/Use"
    categories = [
        'First Party Collection/Use', 'Third Party Sharing/Collection', 'Other',
        'User Choice/Control', 'Do Not Track', 'International and Specific Audiences',
        'Data Security', 'Policy Change', 'Data Retention', 'User Access, Edit and Deletion'
    ]
    prompt_template = """You are a highly specialized privacy policy classifier.
            As an expert in categorizing privacy policy texts your task is to accurately classify the provided text into the most suitable category.
            Additionally, ensure you consider whether multiple categories apply when necessary. Be thorough, and ensure that the output is clear and concise.
            Please use your expertise to categorize the provided privacy policy text into the following categories: First Party Collection/Use, Third Party Sharing/Collection, Other, User Choice/Control, Do Not Track, International and Specific Audiences, Data Security, Policy Change, Data Retention, User Access, Edit and Deletion."""
    
    if model_name == "Fine Tuned Model" or model_name== "Choose an LLM":
        # tokenizer = AutoTokenizer.from_pretrained('/home/bhavani/GA Work/Fine_Tuning_and_ChatBot_Creation/src/FineTuning_11/my-llama2-chat-hf-llm')
        # model = AutoModelForCausalLM.from_pretrained('/home/bhavani/GA Work/Fine_Tuning_and_ChatBot_Creation/src/FineTuning_11/my-llama2-chat-hf-llm', num_labels = 10)
        tokenizer = AutoTokenizer.from_pretrained('/data/bhavani/DeCompressed/FineTuning_11/my-llama2-chat-hf-llm')
        model = AutoModelForCausalLM.from_pretrained('/data/bhavani/DeCompressed/FineTuning_11/my-llama2-chat-hf-llm', num_labels = 10)

    elif model_name =="Fine Tuned Quantized Model":
        quantized_model_path = "/home/bhavani/GA Work/Fine_Tuning_and_ChatBot_Creation/src/Quantized_Models/llama2-quantized_11"
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path,device_map="auto",load_in_8bit=True)

    if model_name =="Base Line Model - LLaMA 2":
        HfFolder.save_token("<Your Hugging Face Token>")
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
        model = pipeline("text-generation", model=model_name, tokenizer=tokenizer,
                                torch_dtype=torch.float16, device_map="auto")
        sentences = nltk.sent_tokenize(pp_text)
        res = ""
        for sentence in sentences:
            prompt = prompt_template + " " + sentence + "\n\nCategory:"
            predicted_text = model(prompt, do_sample=True, top_k=10, num_return_sequences=1,
                                            eos_token_id=tokenizer.eos_token_id, max_length=2060)
            generated_text = predicted_text[0]['generated_text'][len(prompt):].strip()  
            category_text = None
            # print("Generated Text::::",generated_text)
            if "Explanation:" in generated_text:
                category_text = generated_text.split("Explanation:")[0].strip()
            elif "Reason:" in generated_text:
                category_text = generated_text.split("Reason:")[0].strip()
            elif "Reasoning:" in generated_text:
                category_text = generated_text.split("Reasoning:")[0].strip()
            else:
                category_text = generated_text.strip()
            if "Category:" in category_text:
                category_text = category_text.split("Category:")[-1].strip()
            predicted_categories = extract_category(category_text,categories)
            res+="Sentence: "+sentence+" Category: "+category_text+" "
        # print("Done")
        return res
    else:
        sentences = nltk.sent_tokenize(pp_text)
        # print(sentences)
        res = ""
        for sentence in sentences:
            prompt = prompt_template + " " + sentence + "\n\nCategory:"
            inputs = tokenizer.encode(prompt, return_tensors='pt')
            with torch.no_grad():
                output = model.generate(inputs, max_new_tokens=20)  
            predicted_text = tokenizer.decode(output[0], skip_special_tokens=True)

            # print("predicted_text:",predicted_text)        
            generated_text = predicted_text[len(prompt):].strip()  #
            predicted_categories = generated_text.split('Category:')[-1].strip().split(', ')

            print(f"Generated new tokens: {generated_text}")
            print("Sentence:", sentence+"Category:",predicted_categories)
            # print(tokenizer.tokenize(generated_text))
            res+="Sentence: "+sentence+" Category: "+predicted_categories[0]+" "
        # print("Done")
        return res
    
def extract_sentences(text):
    sentences = re.findall(r"Sentence:\s(.*?)\s*Category:", text)
    cleaned_text = " ".join(sentences).strip()
    return cleaned_text


def paraphrase(button_clicked,text_to_be_paraphrased):
    HfFolder.save_token("<Your Hugging Face Token>")
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    model = pipeline("text-generation", model=model_name, tokenizer=tokenizer,
                            torch_dtype=torch.float16, device_map="auto")
    cleaned_text = extract_sentences(text_to_be_paraphrased)
    response = ""
    if button_clicked=="Simplify for Kids":
        prompt = f"""Paraphrase the following privacy policy text into simple, child-friendly language suitable for a 10-year-old. 
                    Make it engaging, easy to understand, and free of complex words while keeping the original meaning. 
                    Imagine you are explaining this to a child who uses an app but doesn’t understand privacy policies. 
                    Your goal is to help them understand how their information is collected and protected in a fun and simple way.
                    You are a friendly teacher or parent explaining privacy in a way a child would understand.
                    Use short and simple sentences, replace complex words with everyday language, and add relatable analogies (e.g., 'Just like a teacher remembers what you like to help you learn better, we remember what you like in the app!").
                    Keep it warm, friendly, and fun. Here is the text to be paraphrased: {cleaned_text} """
        response = "Here is the paraphrased version for the Kids."
        
    elif button_clicked=="Simplify for Adults":
        prompt = f""" Rewrite the following privacy policy text in a clear and simple way for an 80-year-old.
                    Maintain the original meaning but avoid legal and technical jargon, ensuring it's easy to read and understand.
                    Assume the reader is an elderly person who may not be familiar with technology or legal terms.
                    Your goal is to help them understand how their data is collected and used without confusion.
                    You are a patient and caring assistant explaining privacy policies to seniors in a respectful, clear, and professional way.
                    Use short sentences and plain language, avoid technical terms (e.g., replace "data processing" with "how we use your information"), and use real-world comparisons (e.g., "Just like a doctor keeps records to take better care of you, we store some information to improve our services.").
                    Keep it calm, professional, and reassuring. Here is the text to be paraphrased: {cleaned_text} """
        response = "Here is the paraphrased version for the Adults."
        
    elif button_clicked=="Paraphrase":
        prompt = f""" Rewrite the following privacy policy text in clear, plain language while keeping it legally accurate and professional.
                     Simplify complex sentences and improve readability while preserving the original meaning.
                     The audience includes everyday users of a website or app, many of whom may not understand legal terminology.
                     The goal is to ensure clarity and transparency while keeping the policy concise and easy to follow.
                     You are a legal assistant who helps companies communicate privacy policies in clear, user-friendly terms.
                     Keep it concise but legally correct, avoid unnecessary complexity, and use short paragraphs and simple language.
                     Maintain a neutral, professional, and transparent tone. Here is the text to be paraphrased: {cleaned_text} """
        response = "Here is the parapharsed version."
    print(prompt)
    predicted_text =  model(prompt+cleaned_text, do_sample=True, top_k=10, num_return_sequences=1,
                                        eos_token_id=tokenizer.eos_token_id, max_length=2060)
    print("Paraphrase_ content:",predicted_text[0].get("generated_text"))
    paraphrased_text = predicted_text[0].get("generated_text")
    if prompt in paraphrased_text:
        paraphrased_text = paraphrased_text.split(prompt)[1]
        print("text",paraphrased_text)
    return response + paraphrased_text

    

def extract_category(text, list_of_categories):
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, list_of_categories)) + r')\b', re.IGNORECASE)
    matches = pattern.findall(text)
    return matches if matches else ["Unknown Category"]
        
@csrf_exempt
def test(request):
    data = request.body.decode('utf-8')
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    model = pipeline("text-generation", model=model_name, tokenizer=tokenizer,
                                  torch_dtype=torch.float16, device_map="auto")
    
    prompt = "Given a privacy policy text, paraphrase it for a 10 year old kid:" + data
    predicted_text = model(prompt, do_sample=True, top_k=10, num_return_sequences=1,
                                        eos_token_id=tokenizer.eos_token_id, max_length=2060)
    total_tokens = len(tokenizer.encode(predicted_text[0]['generated_text']))  # Count total tokens in the output

    # generated_text = predicted_text[0]['generated_text'][len(prompt):].strip()  # Extract only the generated portion
    hf_token = "<Your Hugging Face Token>"
    HfFolder.save_token(hf_token)
    print("paraphrased text:",predicted_text[0].get("generated_text"))
    return JsonResponse({"message": predicted_text[0].get("generated_text")})
