export type ReviewStatus =
  | 'pending'
  | 'approved'
  | 'edited'
  | 'rejected'
  | 'more_research_requested'

export type ReviewDecision = 'approved' | 'edited' | 'rejected' | 'more_research_requested'

export interface Citation {
  document_extraction_id: number
  page_number: number
  supporting_text: string
}

export interface ReviewSummary {
  id: number
  status: ReviewStatus
  company_number: string
  question: string
  claim_type: 'fact' | 'interpretation'
  evidence_sufficient: boolean
  review_reason: string
  created_at: string
}

export interface ReviewDetail extends ReviewSummary {
  generated_query: string
  claim: string
  citations: Citation[]
  edited_claim: string | null
  decision_note: string | null
  reviewer: string | null
  decided_at: string | null
}

export interface ReviewDecisionRequest {
  decision: ReviewDecision
  edited_claim?: string | null
  note?: string | null
  reviewer?: string | null
}

export interface ReviewDecisionResponse {
  review_id: number
  status: ReviewDecision
  claim: string
}
