class TrustLLmGeneration:
    
    def __init__(self, model, api_key):

        self.model=model
        
        openai_client=OpenAI(api_key=api_key)
        
        ## create a dict of various models and their clients
        self.model_client_map={'gpt-4o':openai_client, 'gpt-3.5-turbo':openai_client, 'gpt-4':openai_client}
        
        ## for each model, a different functon would be called 
        self.llm_client=self.model_client_map[model]

    def get_llmtrust_message(self, prompt):
        system_prompt={'role':'system', 'content':"You are a helpful assistent."}
        user_prompt={'role':'user', 'content': prompt}
    
        return [system_prompt, user_prompt]

    def get_openai_response(self, llm_client, model, messages, temperature=0):
        return llm_client.chat.completions.create(model=model, messages=messages, temperature=temperature)

    def get_llamma_response(self, llamma_client, model, messages, temperature=0):
        ### returns the response from the llama model
        return 
        
    def get_answer_from_llm_response(self, response):
        answer=response.choices[0].message.content.split(":")[1].strip()
        return int(answer)   

    def get_llm_generated_answer(self, prompt):
        
        messages=self.get_llmtrust_message(prompt)
        
        if 'gpt' in self.model:
            resp_function=self.get_openai_response
            
        llm_response=resp_function(self.llm_client, self.model, messages)
        answer=get_answer_from_llm_response(llm_response)

        return answer
        