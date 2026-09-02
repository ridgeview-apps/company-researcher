import { useEffect, useState } from 'react'
import { ApiError, decideReview, fetchReview } from '../api'
import type { ReviewDecision, ReviewDetail } from '../types'

const DECISIONS: Array<{ label: string; value: ReviewDecision }> = [
  { label: 'Approve', value: 'approved' },
  { label: 'Edit claim', value: 'edited' },
  { label: 'Reject', value: 'rejected' },
  { label: 'Request more research', value: 'more_research_requested' },
]

interface ReviewDetailPanelProps {
  reviewId: number
  onBack: () => void
  onDecided: () => void
}

export function ReviewDetailPanel({ reviewId, onBack, onDecided }: ReviewDetailPanelProps) {
  const [review, setReview] = useState<ReviewDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const [decision, setDecision] = useState<ReviewDecision>('approved')
  const [editedClaim, setEditedClaim] = useState('')
  const [note, setNote] = useState('')
  const [reviewer, setReviewer] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    fetchReview(reviewId)
      .then((data) => {
        if (!cancelled) {
          setReview(data)
          setEditedClaim(data.claim)
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof ApiError ? err.message : 'Failed to load review')
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [reviewId])

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault()
    setSubmitting(true)
    setSubmitError(null)
    try {
      await decideReview(reviewId, {
        decision,
        edited_claim: decision === 'edited' ? editedClaim : undefined,
        note: note || undefined,
        reviewer: reviewer || undefined,
      })
      onDecided()
      const refreshed = await fetchReview(reviewId)
      setReview(refreshed)
    } catch (err) {
      setSubmitError(err instanceof ApiError ? err.message : 'Failed to record decision')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="review-detail">
      <button type="button" className="back-link" onClick={onBack}>
        ← Back to reviews
      </button>

      {loading && <p className="muted">Loading…</p>}
      {error && <p className="error">{error}</p>}

      {review && (
        <>
          <header>
            <span className={`badge status-${review.status}`}>{review.status}</span>
            <span className={`badge claim-${review.claim_type}`}>
              {review.claim_type}
            </span>
            <span
              className={`badge ${review.evidence_sufficient ? 'evidence-ok' : 'evidence-insufficient'}`}
            >
              {review.evidence_sufficient ? 'evidence sufficient' : 'evidence insufficient'}
            </span>
          </header>

          <p className="field-label">Company</p>
          <p>{review.company_number}</p>

          <p className="field-label">Question</p>
          <p>{review.question}</p>

          <p className="field-label">Why flagged for review</p>
          <p>{review.review_reason}</p>

          <p className="field-label">Claim</p>
          <p className="claim">{review.claim}</p>

          <p className="field-label">Generated search query</p>
          <p className="mono">{review.generated_query}</p>

          <p className="field-label">Citations</p>
          <ul className="citations">
            {review.citations.map((citation, index) => (
              <li key={index} className="citation">
                <p className="citation-meta">
                  Extraction {citation.document_extraction_id}, page{' '}
                  {citation.page_number}
                </p>
                <p className="citation-text">"{citation.supporting_text}"</p>
              </li>
            ))}
          </ul>

          {review.status === 'pending' ? (
            <form className="decision-form" onSubmit={handleSubmit}>
              <p className="field-label">Decision</p>
              <div className="decision-options">
                {DECISIONS.map((option) => (
                  <label key={option.value}>
                    <input
                      type="radio"
                      name="decision"
                      value={option.value}
                      checked={decision === option.value}
                      onChange={() => setDecision(option.value)}
                    />
                    {option.label}
                  </label>
                ))}
              </div>

              {decision === 'edited' && (
                <>
                  <p className="field-label">Edited claim</p>
                  <textarea
                    value={editedClaim}
                    onChange={(event) => setEditedClaim(event.target.value)}
                    rows={4}
                  />
                </>
              )}

              <p className="field-label">Note (optional)</p>
              <textarea
                value={note}
                onChange={(event) => setNote(event.target.value)}
                rows={2}
              />

              <p className="field-label">Reviewer (optional)</p>
              <input
                type="text"
                value={reviewer}
                onChange={(event) => setReviewer(event.target.value)}
              />

              {submitError && <p className="error">{submitError}</p>}

              <button type="submit" disabled={submitting}>
                {submitting ? 'Submitting…' : 'Submit decision'}
              </button>
            </form>
          ) : (
            <div className="decided-info">
              <p className="field-label">Decided</p>
              {review.edited_claim && (
                <>
                  <p className="field-label">Edited claim</p>
                  <p className="claim">{review.edited_claim}</p>
                </>
              )}
              {review.decision_note && (
                <>
                  <p className="field-label">Note</p>
                  <p>{review.decision_note}</p>
                </>
              )}
              {review.reviewer && (
                <>
                  <p className="field-label">Reviewer</p>
                  <p>{review.reviewer}</p>
                </>
              )}
            </div>
          )}
        </>
      )}
    </div>
  )
}
