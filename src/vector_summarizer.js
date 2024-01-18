import { CheerioWebBaseLoader } from 'langchain/document_loaders/web/cheerio'
import { ChatOpenAI } from 'langchain/chat_models/openai'
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter'
import { OpenAIEmbeddings } from 'langchain/embeddings/openai'
import { MemoryVectorStore } from 'langchain/vectorstores/memory'
  import { RetrievalQAChain } from 'langchain/chains'

const gigantic_page_50k = "https://en.wikipedia.org/wiki/World_War_II"

const llm_16k = 'gpt-3.5-turbo-16k'

const llm = new ChatOpenAI({ modelName: llm_16k })

const loader = new CheerioWebBaseLoader(gigantic_page_50k)
const data = await loader.load()

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 5000,
  chunkOverlap: 1000
})

const splitted = await textSplitter.splitDocuments(data)

const vectorStore = await MemoryVectorStore.fromDocuments(splitted, new OpenAIEmbeddings())

const chain = RetrievalQAChain.fromLLM(llm, vectorStore.asRetriever())

const response = await chain.call({
  query: "Can you provide a summary of the content consisting of an introduction and 10 bullet points?"
})

console.log(response.text)