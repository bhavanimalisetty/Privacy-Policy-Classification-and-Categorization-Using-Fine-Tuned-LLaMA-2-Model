# import json
# from channels.generic.websocket import WebsocketConsumer
# from .views import *
# class MyConsumer(WebsocketConsumer):
#     def connect(self):
#         self.accept()
#         self.send(text_data=json.dumps({
#             'message': 'GeeksforGeeks'
#         }))
    
#     def disconnect(self, close_code):
#         pass
    
#     def receive(self, text_data):
#         text_data_json = json.loads(text_data)
#         print(text_data_json)
#         if "model_name" in text_data_json.keys():
#             privacy_policy_text = text_data_json["privacy_policy_input_text"]
#             model_name = text_data_json["model_name"]
#             pp_classifier_output = llm_response(privacy_policy_text, model_name)
#             self.send(text_data=json.dumps({
#                 "message_from_server": pp_classifier_output
#             }))
#         elif "button_clicked" in text_data_json.keys():
#             button_clicked  = text_data_json["button_clicked"]
#             text_to_be_paraphrased = text_data_json["privacy_policy_output_text"]
#             print(text_to_be_paraphrased)

#             res = paraphrase(button_clicked,text_to_be_paraphrased)
#             print("in consumers.py",res)
#             self.send(text_data = json.dumps({
#                 "paraphrased_content": res
#             }))


import json
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async
from .views import llm_response, paraphrase

class MyConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        await self.send(text_data=json.dumps({
            'message': 'GeeksforGeeks'
        }))
    
    async def disconnect(self, close_code):
        pass

    @sync_to_async
    def run_llm_response(self, text, model_name):
        return llm_response(text, model_name)
    
    @sync_to_async
    def run_paraphrase(self, tone, text):
        return paraphrase(tone, text)

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        print(text_data_json)

        if "model_name" in text_data_json:
            privacy_policy_text = text_data_json["privacy_policy_input_text"]
            model_name = text_data_json["model_name"]

            pp_classifier_output = await self.run_llm_response(privacy_policy_text, model_name)

            await self.send(text_data=json.dumps({
                "message_from_server": pp_classifier_output
            }))

        elif "button_clicked" in text_data_json:
            button_clicked = text_data_json["button_clicked"]
            text_to_be_paraphrased = text_data_json["privacy_policy_output_text"]

            print(text_to_be_paraphrased)

            res = await self.run_paraphrase(button_clicked, text_to_be_paraphrased)
            print("in consumers.py", res)

            await self.send(text_data=json.dumps({
                "paraphrased_content": res
            }))
