import React, { useEffect, useState } from 'react'
import './style.css'

const RECS_API_URL = 'http://localhost:8000/api/recommendations/query/'
const STREAMING_API_BASE = 'http://localhost:8000/api/streaming'

type Recommendation = {
  title: string
  platform: string
  type: string
  listed_in: string
  description: string
  release_year: number | string
  final_score: number
  semantic_score: number
  tfidf_score: number
}

type SearchStats = {
  keyword_results?: number
  semantic_results?: number
  merged_results?: number
  final_results?: number
}

type StreamingAccount = {
  id: number
  service: 'netflix' | 'amazon_prime'
  service_name: string
  username_or_email: string
  profile_name: string | null
  status: 'never_synced' | 'syncing' | 'synced' | 'error'
  last_synced_at: string | null
  last_error: string | null
}

type ActivePage = 'discover' | 'connections'

function App() {
  const [activePage, setActivePage] = useState<ActivePage>('discover')

  // --- Recommendation state ---
  const [query, setQuery] = useState('dark sci-fi series like Black Mirror')
  const [platform, setPlatform] = useState<'all' | 'Netflix' | 'Amazon Prime'>('all')
  const [k, setK] = useState(8)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [results, setResults] = useState<Recommendation[]>([])
  const [stats, setStats] = useState<SearchStats | null>(null)

  // --- Streaming account state ---
  const [accounts, setAccounts] = useState<StreamingAccount[]>([])
  const [connectError, setConnectError] = useState<string | null>(null)
  const [connectingService, setConnectingService] = useState<'netflix' | 'amazon_prime' | null>(null)

  const [netflixEmail, setNetflixEmail] = useState('')
  const [netflixPassword, setNetflixPassword] = useState('')
  const [netflixProfile, setNetflixProfile] = useState('')

  const [primeEmail, setPrimeEmail] = useState('')
  const [primePassword, setPrimePassword] = useState('')
  const [primeProfile, setPrimeProfile] = useState('')

  // --- Helpers ---

  async function fetchAccounts() {
    try {
      const res = await fetch(`${STREAMING_API_BASE}/accounts/`)
      if (!res.ok) return
      const data = (await res.json()) as StreamingAccount[]
      setAccounts(data)
    } catch (e) {
      console.warn('Failed to load streaming accounts', e)
    }
  }

  useEffect(() => {
    fetchAccounts()
  }, [])

  async function handleConnect(service: 'netflix' | 'amazon_prime') {
    setConnectError(null)
    setConnectingService(service)
    try {
      const body =
        service === 'netflix'
          ? {
              service: 'netflix',
              username_or_email: netflixEmail,
              password: netflixPassword,
              profile_name: netflixProfile,
              run_sync: true,
            }
          : {
              service: 'amazon_prime',
              username_or_email: primeEmail,
              password: primePassword,
              profile_name: primeProfile,
              run_sync: true,
            }

      const res = await fetch(`${STREAMING_API_BASE}/connect/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })

      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.detail || 'Failed to connect and sync account')
      }

      await fetchAccounts()
    } catch (e: any) {
      console.error(e)
      setConnectError(e.message || 'Failed to connect the streaming account.')
    } finally {
      setConnectingService(null)
    }
  }

  async function handleSearch(e: React.FormEvent) {
    e.preventDefault()
    if (!query.trim()) return

    setLoading(true)
    setError(null)

    try {
      const response = await fetch(RECS_API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query.trim(),
          platform,
          k,
        }),
      })

      if (!response.ok) {
        const body = await response.json().catch(() => ({}))
        throw new Error(body.detail || `Request failed with status ${response.status}`)
      }

      const data = (await response.json()) as {
        results: Recommendation[]
        search_stats?: SearchStats
      }

      setResults(data.results || [])
      setStats(data.search_stats || null)
    } catch (err: any) {
      console.error(err)
      setError(err.message || 'Something went wrong while fetching recommendations.')
    } finally {
      setLoading(false)
    }
  }

  function renderStatusBadge(account?: StreamingAccount) {
    if (!account) return <span className="status-pill status-pill--disconnected">Not connected</span>

    if (account.status === 'syncing') {
      return <span className="status-pill status-pill--syncing">Syncing…</span>
    }
    if (account.status === 'synced') {
      return <span className="status-pill status-pill--synced">Synced</span>
    }
    if (account.status === 'error') {
      return <span className="status-pill status-pill--error">Error</span>
    }
    return <span className="status-pill">Not synced</span>
  }

  const netflixAccount = accounts.find((a) => a.service === 'netflix')
  const primeAccount = accounts.find((a) => a.service === 'amazon_prime')

  // --- Render ---

  return (
    <div className="app-root">
      <header className="app-header">
        <div className="brand">
          <span className="brand-mark" />
          <span className="brand-text">StreamSense</span>
        </div>

        <nav className="nav-tabs">
          <button
            type="button"
            className={
              'nav-tab' + (activePage === 'discover' ? ' nav-tab--active' : '')
            }
            onClick={() => setActivePage('discover')}
          >
            Discover
          </button>
          <button
            type="button"
            className={
              'nav-tab' + (activePage === 'connections' ? ' nav-tab--active' : '')
            }
            onClick={() => setActivePage('connections')}
          >
            Connections
          </button>
        </nav>
      </header>

      <main className="app-main">
        {activePage === 'connections' ? (
          <section className="connections-page">
            <h1 className="page-title">Manage streaming connections</h1>
            <p className="page-subtitle">
              Connect your Netflix and Prime accounts locally so StreamSense can build a private taste
              profile from your viewing history.
            </p>

            {connectError && <div className="error-banner error-banner--wide">{connectError}</div>}

            <div className="connections-grid">
              <div className="connection-card">
                <div className="connection-header">
                  <h2>Connect Netflix</h2>
                  {renderStatusBadge(netflixAccount)}
                </div>
                <p className="connection-help">
                  Credentials are only used on this machine via Selenium to pull your viewing history.
                  They are stored encrypted for this demo.
                </p>
                <div className="connection-form">
                  <input
                    className="connection-input"
                    type="text"
                    placeholder="Email or username"
                    value={netflixEmail}
                    onChange={(e) => setNetflixEmail(e.target.value)}
                  />
                  <input
                    className="connection-input"
                    type="password"
                    placeholder="Password"
                    value={netflixPassword}
                    onChange={(e) => setNetflixPassword(e.target.value)}
                  />
                  <input
                    className="connection-input"
                    type="text"
                    placeholder="Profile name (as shown in Netflix)"
                    value={netflixProfile}
                    onChange={(e) => setNetflixProfile(e.target.value)}
                  />
                  <button
                    type="button"
                    className="connection-button"
                    onClick={() => handleConnect('netflix')}
                    disabled={connectingService === 'netflix'}
                  >
                    {connectingService === 'netflix' ? 'Syncing…' : 'Save & Sync'}
                  </button>
                </div>
              </div>

              <div className="connection-card">
                <div className="connection-header">
                  <h2>Connect Prime Video</h2>
                  {renderStatusBadge(primeAccount)}
                </div>
                <p className="connection-help">
                  Sync your Prime watch history to power cross-platform recommendations.
                </p>
                <div className="connection-form">
                  <input
                    className="connection-input"
                    type="text"
                    placeholder="Email or username"
                    value={primeEmail}
                    onChange={(e) => setPrimeEmail(e.target.value)}
                  />
                  <input
                    className="connection-input"
                    type="password"
                    placeholder="Password"
                    value={primePassword}
                    onChange={(e) => setPrimePassword(e.target.value)}
                  />
                  <input
                    className="connection-input"
                    type="text"
                    placeholder="Profile name (if applicable)"
                    value={primeProfile}
                    onChange={(e) => setPrimeProfile(e.target.value)}
                  />
                  <button
                    type="button"
                    className="connection-button"
                    onClick={() => handleConnect('amazon_prime')}
                    disabled={connectingService === 'amazon_prime'}
                  >
                    {connectingService === 'amazon_prime' ? 'Syncing…' : 'Save & Sync'}
                  </button>
                </div>
              </div>
            </div>
          </section>
        ) : (
          <>
            <section className="hero">
              <h1>Find your next favorite movie or series.</h1>
              <p>
                Hybrid recommendations across Netflix and Prime, personalized using your viewing
                taste profile.
              </p>

              <form className="search-form" onSubmit={handleSearch}>
                <div className="search-input-row">
                  <input
                    className="search-input"
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="e.g. slow-burn thrillers like Mindhunter"
                  />
                  <button className="search-button" type="submit" disabled={loading}>
                    {loading ? 'Searching…' : 'Get recommendations'}
                  </button>
                </div>

                <div className="search-filters">
                  <div className="filter-group">
                    <label className="filter-label">Platform</label>
                    <div className="pill-row">
                      {(['all', 'Netflix', 'Amazon Prime'] as const).map((plt) => (
                        <button
                          key={plt}
                          type="button"
                          className={
                            'pill' + (platform === plt ? ' pill--active' : '')
                          }
                          onClick={() => setPlatform(plt)}
                        >
                          {plt === 'all' ? 'All platforms' : plt}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="filter-group">
                    <label className="filter-label">How many titles?</label>
                    <input
                      className="k-input"
                      type="number"
                      min={1}
                      max={20}
                      value={k}
                      onChange={(e) => setK(Number(e.target.value) || 5)}
                    />
                  </div>
                </div>
              </form>

              {error && <div className="error-banner">{error}</div>}
              {stats && (
                <div className="stats-row">
                  <span>
                    Keyword: <strong>{stats.keyword_results ?? 0}</strong>
                  </span>
                  <span>
                    Semantic: <strong>{stats.semantic_results ?? 0}</strong>
                  </span>
                  <span>
                    Final picks: <strong>{stats.final_results ?? results.length}</strong>
                  </span>
                </div>
              )}
            </section>

            <section className="results-section">
              {loading && <p className="subtle-text">Crunching embeddings and TF-IDF…</p>}
              {!loading && results.length === 0 && !error && (
                <p className="subtle-text">Try a query to see personalized recommendations.</p>
              )}

              <div className="results-grid">
                {results.map((item) => (
                  <article key={`${item.platform}-${item.title}-${item.release_year}`} className="result-card">
                    <div className="card-header">
                      <h2>{item.title}</h2>
                      <span className={`platform-badge platform-${item.platform.replace(' ', '').toLowerCase()}`}>
                        {item.platform}
                      </span>
                    </div>
                    <div className="card-meta">
                      <span>{item.type}</span>
                      {item.release_year && <span>· {item.release_year}</span>}
                    </div>
                    {item.listed_in && (
                      <div className="card-genres">
                        {item.listed_in.split(',').slice(0, 3).map((g) => (
                          <span key={g.trim()} className="genre-chip">
                            {g.trim()}
                          </span>
                        ))}
                      </div>
                    )}
                    {item.description && (
                      <p className="card-description">
                        {item.description.length > 180
                          ? item.description.slice(0, 177) + '...'
                          : item.description}
                      </p>
                    )}

                    <div className="score-row">
                      <span className="score-pill">
                        Hybrid score: {item.final_score.toFixed(3)}
                      </span>
                      <span className="score-sub">
                        kw {item.tfidf_score.toFixed(2)} · sem {item.semantic_score.toFixed(2)}
                      </span>
                    </div>
                  </article>
                ))}
              </div>
            </section>
          </>
        )}
      </main>

      <footer className="app-footer">
        <span>Built on FAISS, TF‑IDF and Gemini. Local demo only.</span>
      </footer>
    </div>
  )
}

export default App
