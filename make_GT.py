# 각 청크에서 Q&A 형식의 GroundTruth 생성

def main():
    # 필요한 라이브러리 임포트
    from langchain_core.documents import Document
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_openai import ChatOpenAI
    from langchain.prompts import PromptTemplate
    from pydantic import BaseModel, Field
    import json
    import os
    from dotenv import load_dotenv

    load_dotenv(override=True)

    # 환경 변수에서 파일 경로 가져오기
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    if os.getenv("OPENAI_API_BASE"):
        os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 벡터 스토어에서 청크 가져오기
    try:
        faiss_store = FAISS.load_local(
            "faiss_store",
            embeddings,
            allow_dangerous_deserialization=True,  # 신뢰할 수 있는 파일만 True로!
        )
        print("✅ faiss_store를 성공적으로 불러왔습니다.")
    except Exception as e:
        print(f"❌문제 발생 : {e}")
        print("faiss_store를 불러오는 데 실패 파일 경로와 임베딩 모델을 확인")
        return

    try:
        chunks = list(faiss_store.docstore._dict.values())
        print(f"✅총 {len(chunks)}개의 청크를 불러왔습니다.")
    except Exception as e:
        print(f"❌문제 발생 : {e}")
        print("청크를 불러오는 데 실패. faiss_store가 올바르게 로드되었는지 확인")
        return

    # output parser 설정
    class GroundTrunth(BaseModel):
        Q: str = Field(
            description="질문: 짧고 명확한 사실에 대한 형태의 쿼리 (예: 그림을 그리는 시간이 짧았다.) "
        )
        A: str = Field(
            description="답변: context를 읽고 풍부하고 전문가적인 응답을 생성해야 합니다. 단순히 문맥 context를 읽는 것이 아닌 전문가가 해당 특징에 대해 서술하듯이 이야기 해야합니다. context와 해석 여지를 제공하여 평가하기에 용이한 응답을 생성하세요.절대 거짓이나 모호한 응답을 생성해서는 안됩니다. context를 이해하고 작성해주세요."
        )

    parser = JsonOutputParser(pydantic_object=GroundTrunth)

    # 프롬프트 템플릿: 각 청크에서 질문-응답 쌍을 생성하도록 명시
    prompt = PromptTemplate(
        template="""
    사전 정의 : 당신은 HTP(Home, tree, person) 심리 검사에 대한 전문가입니다. 아래에 주어지는 문서를 확인하고 RAG를 구축하고 평가하기 위한 GroundTruth를 생성해야 합니다.
    구축하고자 하는 RAG 파이프라인은 상담사가 그림을 관찰하고 관찰한 그림에 대한 사실적 질문을 생성하여 그에 대한 해석 결과를 제공하는 것입니다.
    (예시)
    예를 들어 상담사는 그림을 관찰하고 "집이 오른쪽으로 기울어져 있다. 문이 집의 크기에 비해 작게 그려져 있고, 지붕은 존재하고 있으나 굴뚝을 그리지 않았다." 라고 작성할 수 있습니다.
    그러면 질문을 "집이 오른쪽으로 기울어져 있다", "문이 집의 크기에 비해 작다", "굴뚝이 없다" 와 같이 각 특이사항 별로 쿼리를 분해하고 RAG에서 검색을 할 수 있도록 합니다.
    각 쿼리에 대한 해석을 정리, 종합하여 LLM 에게 전달하고 LLM은 객관적인 RAG 기반 응답을 생성합니다.

    예시와 같은 서비스를 구축하기 위해 RAG 평가를 위한 GroundTruth를 생성해야하는게 당신의 업무입니다.
    (질문)
    질문을 생성할 때에는 마치 그림을 보고 관찰한 결과를 보고하듯 작성해야 합니다. 가상의 그림이 있다고 생각하고 작성하세요.
    즉 context의 내용으로만 질문을 작성하지 말고 마치 그림이 있다고 생각하고 관찰한 결과를 작성해야 합니다.
    (응답)
    질문에 대한 응답을 생성할 때에는 context를 읽고 풍부하고 전문가적인 응답을 생성해야 합니다. 단순히 문맥 context를 읽는 것이 아닌 전문가가 해당 특징에 대해 서술하듯이 이야기 해야합니다.
    context와 해석 여지를 제공하여 평가하기에 용이한 응답을 생성하세요. 절대 거짓이나 모호한 응답을 생성해서는 안됩니다. context를 이해하고 작성해주세요.

    아래의 문서 내용을 바탕으로, 의문문이 아닌 팩트 기반의 진술문을 쿼리와 답변 쌍을 하나만 생성하세요.
    \n{format_instructions}\n

    문서 내용:
    \n{context}\n

    """,
        input_variables=["context"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    # chain 생성
    chain = prompt | llm | parser

    print("✅체인 생성 완료, GT 생성 시작")

    ground_truths = []

    try:
        for idx, doc in enumerate(chunks):
            context = doc.page_content
            result = chain.invoke({"context": context})
            ground_truths.append(
                {
                    "chunk_id": idx,
                    "main_topic": doc.metadata.get("main_topic"),
                    "sub_topic": doc.metadata.get("sub_topic"),
                    "qa_pairs": result,
                }
            )
            print(f"✅Processed chunk {idx + 1}/{len(chunks)}")
    except Exception as e:
        print(f"❌문제 발생 : {e}")
        return
    
    # ground_truths를 JSON 파일로 저장
    with open("ground_truths.json", "w", encoding="utf-8") as f:
        json.dump(ground_truths, f, ensure_ascii=False, indent=2)

    print("✅ ground_truths.json 파일로 저장 완료")
    print("done")

if __name__ == "__main__":
    main()
