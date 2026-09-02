import type {
  ReviewDecisionRequest,
  ReviewDecisionResponse,
  ReviewDetail,
  ReviewStatus,
  ReviewSummary,
} from './types'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

export class ApiError extends Error {
  status: number

  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

async function parseErrorDetail(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: string }
    return body.detail ?? response.statusText
  } catch {
    return response.statusText
  }
}

export async function fetchReviews(status?: ReviewStatus): Promise<ReviewSummary[]> {
  const url = new URL('/reviews', API_BASE_URL)
  if (status) {
    url.searchParams.set('status', status)
  }
  const response = await fetch(url)
  if (!response.ok) {
    throw new ApiError(await parseErrorDetail(response), response.status)
  }
  return (await response.json()) as ReviewSummary[]
}

export async function fetchReview(reviewId: number): Promise<ReviewDetail> {
  const response = await fetch(new URL(`/reviews/${reviewId}`, API_BASE_URL))
  if (!response.ok) {
    throw new ApiError(await parseErrorDetail(response), response.status)
  }
  return (await response.json()) as ReviewDetail
}

export async function decideReview(
  reviewId: number,
  body: ReviewDecisionRequest,
): Promise<ReviewDecisionResponse> {
  const response = await fetch(new URL(`/reviews/${reviewId}/decision`, API_BASE_URL), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!response.ok) {
    throw new ApiError(await parseErrorDetail(response), response.status)
  }
  return (await response.json()) as ReviewDecisionResponse
}
