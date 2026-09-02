import { useState } from 'react'
import { ReviewDetailPanel } from './components/ReviewDetailPanel'
import { ReviewList } from './components/ReviewList'

function App() {
  const [selectedReviewId, setSelectedReviewId] = useState<number | null>(null)
  const [refreshKey, setRefreshKey] = useState(0)

  return (
    <div className="app">
      <h1>Company Researcher — Analyst Review</h1>
      {selectedReviewId === null ? (
        <ReviewList onSelect={setSelectedReviewId} refreshKey={refreshKey} />
      ) : (
        <ReviewDetailPanel
          reviewId={selectedReviewId}
          onBack={() => setSelectedReviewId(null)}
          onDecided={() => setRefreshKey((key) => key + 1)}
        />
      )}
    </div>
  )
}

export default App
