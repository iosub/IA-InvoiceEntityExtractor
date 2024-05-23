import streamlit as st
from tempfile import NamedTemporaryFile
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


def main():
    st.title("Invoice Entity Extractor:books:")

    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        loader = PyPDFLoader(temp_file_path)
        pages = loader.load()

        st.write(f"Number of pages: {len(pages)}")

        for page in pages:
            st.write(page.page_content)
#https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/README.md?code=true

#https://github.com/marella/ctransformers
#https://python.langchain.com/v0.1/docs/integrations/providers/ctransformers/

#patliu1001/llama-2-7b-chat.ggmlv3.q8_0.bin
#https://huggingface.co/BashitAli/llama-2-7b-chat.ggmlv3.q5_K_M/tree/main
# Load model directly
#from transformers import AutoModel
#model = AutoModel.from_pretrained("BashitAli/llama-2-7b-chat.ggmlv3.q5_K_M")
#BashitAli/llama-2-7b-chat.ggmlv3.q5_K_M
      #  llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0e.bin",model_type="llama",
      #https://huggingface.co/atharvapawar/llama-2-7b-chat.ggmlv3.q8_0/tree/main
      #atharvapawar/llama-2-7b-chat.ggmlv3.q8_0
      
        llm = CTransformers(model="atharvapawar/llama-2-7b-chat.ggmlv3.q8_0.bin",model_type="llama",
             config={'max_new_tokens':128,'temperature':0.01})
        
        template = """Extract invoice number, name of organization, address, date, 
            Qty, Rate ,Tax ,Amount {pages}
        Output : entity : type
        """
        prompt_template = PromptTemplate(input_variables=["pages"], template=template)
        chain = LLMChain(llm=llm, prompt=prompt_template)

        result = chain.run(pages=pages[0].page_content)
        
        st.write("Extracted entities:")
        entities = result.strip().split("\n")
        table_data = [line.split(":") for line in entities]
        st.table(table_data)

if __name__ == "__main__":
    main()



