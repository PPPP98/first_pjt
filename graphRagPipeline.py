def main():
    from dotenv import load_dotenv
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    import os
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever
    from typing import List, TypedDict
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
    from langgraph.graph import END, StateGraph

    load_dotenv()

    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE")

    llm = ChatOpenAI(model="gpt-4o-mini")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    faiss_store = (
        FAISS.load_local(
            "faiss_store", embeddings=embeddings, allow_dangerous_deserialization=True
        )
        if os.path.exists("faiss_store")
        else None
    )
    all_texts = [doc.page_content for doc in faiss_store.docstore._dict.values()]

    K = 3
    weights = [0.3, 0.7]

    # Sparse
    sparse_bm25_retriever = BM25Retriever.from_texts(texts=all_texts)
    sparse_bm25_retriever.k = K  # k값을 통일하여 설정
    # Dense
    dense_similarity_retriever = faiss_store.as_retriever(
        search_type="similarity", search_kwargs={"k": K}
    )

    retriever = EnsembleRetriever(
        retrievers=[sparse_bm25_retriever, dense_similarity_retriever],
        weights=weights,
    )

    class GraphState(TypedDict):
        """
        Graph RAG 파이프라인의 상태

        Attributes:
            original_question (str): 사용자가 입력한 원본 질문 (그림 묘사)
            decomposed_questions (List[str]): 의미 단위로 분해된 질문 리스트
            retrieved_contexts (List[str]): 검색된 관련 문서(해석) 조각 리스트
            generation (str): LLM이 생성한 최종 해석
            relevance_check (str): 질문의 HTP 검사 관련성 여부 ("yes" or "no")
            hallucination_check (str): 생성된 답변의 환각 현상 유무 ("yes" or "no")
            category (str): 질문 범주(집, 나무, 사람)
        """
        original_question: str
        decomposed_questions: List[str]
        retrieved_contexts: List[str]
        generation: str
        relevance_check: str
        hallucination_check: str
        category: str


    def relevance_check_node(state: GraphState):
        """
        입력된 질문이 HTP 심리검사 해석과 관련이 있는지 확인합니다.
        """
        print("--- 1. 질문 관련성 검사 시작 ---")
        question = state["original_question"]
        
        prompt = ChatPromptTemplate.from_template(
            """당신은 심리검사 전문가입니다. 주어진 질문이 'HTP(집-나무-사람) 그림 심리검사' 해석과 관련된 내용인지 판단해주세요.
            HTP 검사는 집, 나무, 사람 그림의 특징(예: 지붕, 문, 창문, 나무 기둥, 가지, 사람의 눈, 코, 입 등)을 분석하는 것입니다.
            질문은 HTP 그림에 대한 관찰 묘사로, 그림의 요소나 특징에 대한 내용입니다.
            이에 해당하면 'yes', 전혀 관련 없는 내용(예: 오늘 날씨, 스포츠 경기 결과 등)이면 'no'로만 대답해주세요.

            질문: {question}
            판단 (yes/no):"""
        )
        
        chain = prompt | llm | StrOutputParser()
        relevance = chain.invoke({"question": question})
        
        print(f"질문 관련성: {relevance}")
        state["relevance_check"] = relevance.lower()
        if state["relevance_check"] == "no":
            # print("질문이 HTP 검사와 관련이 없습니다. 프로세스를 종료합니다.")
            state["generation"] = "관찰 결과를 다시 입력해주세요."
            print(state)
        return state

    def decompose_query_node(state: GraphState):
        """
        입력된 질문을 의미 단위의 여러 하위 질문으로 분해합니다.
        """
        print("--- 2. 질문 분해 시작 ---")
        question = state["original_question"]
        
        prompt = ChatPromptTemplate.from_template(
            """당신은 HTP 그림 심리검사 문장 분석 전문가입니다. 상담사가 그림을 보고 관찰한 내용을 나열한 문장이 주어집니다. 
            이 문장을 그림의 각 요소(예: 문, 창문, 지붕, 길 등)에 대한 독립적인 해석이 가능한 단위로 분해하여 JSON 리스트 형태로 반환해주세요.

            예시:
            입력: "집에 창문은 2개 존재하고 크기는 적절함. 문은 집의 크기에 비해 작으며 문과 바깥이 길로 이어져 있지 않음."
            출력: {{"queries": ["집 창문의 개수는 2개이고 크기는 적절하다.", "집 문의 크기가 집 전체에 비해 작다.", "집과 외부를 잇는 길이 그려져 있지 않다."]}}

            입력: {question}
            출력:"""
        )
        
        chain = prompt | llm | JsonOutputParser()
        decomposed = chain.invoke({"question": question})
        
        decomposed_questions = decomposed.get("queries", [])
        print(f"분해된 질문: {decomposed_questions}")
        state["decomposed_questions"] = decomposed_questions
        return state

    # 2-3. Retrieve Node (정보 검색)
    def retrieve_node(state: GraphState):
        """
        분해된 각 질문에 대해 Ensemble Retriever를 사용하여 관련 문서를 검색합니다.
        """
        print("--- 3. 정보 검색 시작 ---")
        decomposed_questions = state["decomposed_questions"]
        all_retrieved_docs = []

        for query in decomposed_questions:
            print(f"  - 검색 쿼리: '{query}'")
            # 여기서 사용자가 제공한 retriever를 사용합니다.
            retrieved_docs = retriever.invoke(query)
            
            # 검색 결과의 내용을 문자열로 변환하여 추가
            doc_texts = [doc.page_content for doc in retrieved_docs]
            all_retrieved_docs.extend(doc_texts)
        
        # 중복 제거
        unique_contexts = list(set(all_retrieved_docs))
        print(f"검색된 해석 Context 수: {len(unique_contexts)}")
        state["retrieved_contexts"] = unique_contexts
        return state

    # 2-4. Generate Node (답변 생성)
    def generate_node(state: GraphState):
        """
        검색된 Context를 바탕으로 최종 해석 답변을 생성합니다.
        """
        print("--- 4. 답변 생성 시작 ---")
        question = state["original_question"]
        contexts = "\n\n".join(state["retrieved_contexts"])
        
        prompt = ChatPromptTemplate.from_template(
            """당신은 HTP 그림 심리검사 결과 해석 전문가입니다.
            상담사가 관찰한 '그림 특징'과 그에 대한 '해석 참고자료'가 주어집니다. 
            두 정보를 종합하여, 내담자의 심리상태에 대한 최종 해석 보고서를 작성해주세요.
            전문적이고 이해하기 쉬운 말투로 설명하고, 각 특징과 해석을 논리적으로 연결하여 설명해주세요.

            ## 상담사의 그림 특징 관찰 내용:
            {question}

            ## 해석 참고자료:
            {context}

            ## 최종 해석 보고서:
            """
        )
        
        chain = prompt | llm | StrOutputParser()
        generation = chain.invoke({"question": question, "context": contexts})
        
        print("생성된 답변 일부:", generation[:200] + "...")
        state["generation"] = generation
        return state

    # 2-5. Hallucination Check Node (환각 검사)
    def hallucination_check_node(state: GraphState):
        """
        생성된 답변에 환각(hallucination)이 있는지 검사합니다.
        """
        print("--- 5. 환각 검사 시작 ---")
        contexts = state["retrieved_contexts"]
        generation = state["generation"]
        
        prompt = ChatPromptTemplate.from_template(
            """당신은 AI 답변 검증 전문가입니다. 주어진 '참고 자료'를 바탕으로 '생성된 답변'이 만들어졌는지 확인해야 합니다.
            '생성된 답변'의 모든 내용이 '참고 자료'에 근거하고 있다면 'yes'를, '참고 자료'에 없는 내용이 포함되어 있다면 'no'를 반환해주세요.

            ## 참고 자료:
            {context}

            ## 생성된 답변:
            {generation}

            판단 (yes/no):"""
        )
        
        chain = prompt | llm | StrOutputParser()
        check_result = chain.invoke({"context": "\n\n".join(contexts), "generation": generation})

        print(f"환각 검사 결과: {check_result}")
        state["hallucination_check"] = check_result.lower()
        return state
    
    def decide_after_relevance_check(state: GraphState):
        """
        질문 관련성 검사 결과에 따라 다음 단계를 결정합니다.
        - "yes": 질문 분해 단계로 이동
        - "no": 종료
        """
        if state["relevance_check"] == "yes":
            print("결과: 관련성 있음. 질문 분해를 진행합니다.")
            return "decompose"
        else:
            print("결과: 관련성 없음. 프로세스를 종료합니다.")
            return "end"

    # 3-2. 환각 검사 후 분기
    def decide_after_hallucination_check(state: GraphState):
        """
        환각 검사 결과에 따라 다음 단계를 결정합니다.
        - "yes": 환각 없음, 종료
        - "no": 환각 존재, 답변 재생성
        """
        if state["hallucination_check"] == "yes":
            print("결과: 환각 없음. 최종 답변을 반환합니다.")
            return "end"
        else:
            print("결과: 환각 존재. 답변을 다시 생성합니다.")
            return "regenerate"
        

    # 그래프 빌더 생성
    workflow = StateGraph(GraphState)

    # 노드 추가
    workflow.add_node("relevance_check", relevance_check_node)
    workflow.add_node("decompose_query", decompose_query_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("hallucination_check", hallucination_check_node)

    # 엣지 연결
    workflow.set_entry_point("relevance_check")
    workflow.add_conditional_edges(
        "relevance_check",
        decide_after_relevance_check,
        {"decompose": "decompose_query", "end": END}
    )
    workflow.add_edge("decompose_query", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "hallucination_check")
    workflow.add_conditional_edges(
        "hallucination_check",
        decide_after_hallucination_check,
        {"regenerate": "generate", "end": END}
    )

    # 그래프 컴파일
    app = workflow.compile()

    # # HTP 그림 검사 예시 질문
    # inputs = {
    #     "original_question": "집에 창문은 2개 존재하고 크기는 적절함. 문은 집의 크기에 비해 작으며 문과 바깥이 길로 이어져 있지 않음."
    # }

    # # # 그래프 실행
    # # for output in app.stream(inputs, {"recursion_limit": 5}): # 순환 방지를 위해 recursion_limit 설정
    # #     for key, value in output.items():
    # #         # 각 단계의 최종 출력만 표시
    # #         print(f"노드 '{key}' 완료:")
    # #         # print(f"  - 상태: {value}") # 전체 상태를 보려면 주석 해제
    # #         print("---")

    # # 최종 결과 확인
    # final_state = app.invoke(inputs, {"recursion_limit": 6})
    # print("\n" + "="*50)
    # print("          최종 HTP 심리검사 해석 결과")
    # print("="*50)
    # print(final_state['generation'])
    # print("="*50)

if __name__ == "__main__":
    main()