class LLMResponse:
    
    def __init__(self, model, api_key):

        self.model=model
        
        openai_client=OpenAI(api_key=api_key)
        
        ## create a dict of various models and their clients
        self.model_client_map={'gpt-4o':openai_client, 'gpt-3.5-turbo':openai_client, 'gpt-4':openai_client}
        
        ## for each model, a different functon would be called 
        self.llm_client=self.model_client_map[model]

    def get_llm_message(self, question):
        
        '''
        
        docstring: given the question, returns the message as to be used by LLM for generating a response.
        
        '''

        system_prompt={'role':'system', 'content':"""You are an expert in answering common sense multiple choice questions. Your task is to answer a question using common sense as criteria. You will be given two examples for which questions and answer will be given. You can use these examples as a reference to better contextualize as how to answer a question using common sense. Note that this is multiple choice question of which only one answer is correct.\n\n
                                                 The output of your response should be the number corresponding to the selected answer option without any reasons or explanation. \n\n
    
                                                 Here are two examples with questions and answers.
    
                                                 *** Examples ***
    
                                                  ## Example 1 ## 
                                                 
                                                  --- Question : ---
                                                  Question: The professional golfer went to the course to practice. He,
                                                  
                                                     0. putted well.
                                                     1. practiced putting away the green cart.
                                                     2. practiced basketball.
                                                     3. shot a little birdie.
                                                  
                                                  --- Answer: 0 --
    
                                                  ## Example 2 ## 
                                                 
                                                  --- Question : ---
                                                  
                                                  The dog chased the rabbit. The rabbit 
                                                  
                                                     0. got a new identity.
                                                     1. ate the dog.
                                                     2. fled the country.
                                                     3. died.
    
                                                  
                                                  --- Answer: 3 --
          
                                                """}
        
        user_prompt={'role':'user', 'content':f"""Here is the question for you:
    
                                                ** Question: **
                                                {question}
    
                                                Use the following output format to answer your question.
    
                                                *** Output Format ***
                                                answer : <""your_answer"">
    
                                                Where <""your_answer""> is the number corresponding to your answer choice without any explanation or reason.
    
                                                """}
    
        messages=[system_prompt, user_prompt]
    
        return messages
    
    def get_openai_response(self, llm_client, model, messages, temperature=0):
        return llm_client.chat.completions.create(model=model, messages=messages, temperature=temperature)

    def get_llamma_response(self, llamma_client, model, messages, temperature=0):
        ### returns the response from the llama model
        return 
        
    def get_answer_from_llm_response(self, response):
        answer=response.choices[0].message.content.split(":")[1].strip()
        return int(answer)   

    def get_llm_generated_answer(self, question):
        
        messages=self.get_llm_message(question)
        
        if 'gpt' in self.model:
            resp_function=self.get_openai_response
            
        llm_response=resp_function(self.llm_client, self.model, messages)
        answer=get_answer_from_llm_response(llm_response)

        return answer
        