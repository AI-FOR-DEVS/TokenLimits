import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter'
import { CheerioWebBaseLoader } from 'langchain/document_loaders/web/cheerio'
import { loadSummarizationChain } from 'langchain/chains'
import { OpenAI } from 'langchain/llms/openai'

const huge_page_10k = 'https://lilianweng.github.io/posts/2023-06-23-agent/'

const llm_16k = 'gpt-3.5-turbo-16k'

const llm = new OpenAI({ modelName: llm_16k })

const loader = new CheerioWebBaseLoader(huge_page_10k)
const data = await loader.load()

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 100,
})

const splitted = await textSplitter.splitDocuments(data)

const chain = loadSummarizationChain(llm, { type: 'map_reduce' })
const result = await chain.call({
  input_documents: splitted,
})

console.log({ result })
