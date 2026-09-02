import { useEffect, useState } from 'react'
import { ApiError, fetchReviews } from '../api'
import type { ReviewStatus, ReviewSummary } from '../types'

const STATUS_FILTERS: Array<{ label: string; value: ReviewStatus | 'all' }> = [
  { label: 'Pending', value: 'pending' },
  { label: 'Approved', value: 'approved' },
  { label: 'Edited', value: 'edited' },
  { label: 'Rejected', value: 'rejected' },
  { label: 'More research requested', value: 'more_research_requested' },
  { label: 'All', value: 'all' },
]

interface ReviewListProps {
  onSelect: (reviewId: number) => void
  refreshKey: number
}

export function ReviewList({ onSelect, refreshKey }: ReviewListProps) {
  const [statusFilter, setStatusFilter] = useState<ReviewStatus | 'all'>('pending')
  const [reviews, setReviews] = useState<ReviewSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    fetchReviews(statusFilter === 'all' ? undefined : statusFilter)
      .then((data) => {
        if (!cancelled) setReviews(data)
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof ApiError ? err.message : 'Failed to load reviews')
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [statusFilter, refreshKey])

  return (
    <div className="review-list">
      <div className="filter-bar">
        {STATUS_FILTERS.map((filter) => (
          <button
            key={filter.value}
            type="button"
            className={filter.value === statusFilter ? 'filter active' : 'filter'}
            onClick={() => setStatusFilter(filter.value)}
          >
            {filter.label}
          </button>
        ))}
      </div>

      {loading && <p className="muted">Loading…</p>}
      {error && <p className="error">{error}</p>}
      {!loading && !error && reviews.length === 0 && (
        <p className="muted">No reviews with this status.</p>
      )}

      <ul className="review-rows">
        {reviews.map((review) => (
          <li key={review.id}>
            <button
              type="button"
              className="review-row"
              onClick={() => onSelect(review.id)}
            >
              <div className="review-row-badges">
                <span className={`badge status-${review.status}`}>{review.status}</span>
                <span className={`badge claim-${review.claim_type}`}>
                  {review.claim_type}
                </span>
              </div>
              <div className="review-row-main">
                <span className="company">{review.company_number}</span>
                <span className="question">{review.question}</span>
              </div>
              <span className="reason">{review.review_reason}</span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}
