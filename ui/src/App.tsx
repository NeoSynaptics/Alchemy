import { useEffect, lazy, Suspense } from 'react'
import { HashRouter, Routes, Route, NavLink, useLocation } from 'react-router-dom'
import { TooltipProvider } from '@/components/ui/tooltip'
import { useHealth } from '@/hooks/useAlchemy'

const DashboardPage = lazy(() => import('@/pages/Dashboard').then(m => ({ default: m.DashboardPage })))
const BrainPhysicsPage = lazy(() => import('@/pages/BrainPhysics').then(m => ({ default: m.BrainPhysicsPage })))
const SettingsPage = lazy(() => import('@/pages/Settings').then(m => ({ default: m.SettingsPage })))
const MemoryPage = lazy(() => import('@/pages/Memory').then(m => ({ default: m.MemoryPage })))

function NavItem({ to, children }: { to: string; children: React.ReactNode }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `px-3 py-2 text-sm font-medium rounded-md transition-colors ${
          isActive
            ? 'bg-primary text-primary-foreground'
            : 'text-muted-foreground hover:text-foreground hover:bg-muted'
        }`
      }
    >
      {children}
    </NavLink>
  )
}

function HealthDot() {
  const { data: health, error } = useHealth()
  const online = !error && health?.status === 'ok'
  return (
    <div
      className={`h-2.5 w-2.5 rounded-full transition-colors ${
        online
          ? 'bg-emerald-500 shadow-[0_0_6px_rgba(16,185,129,0.5)]'
          : 'bg-red-500 shadow-[0_0_6px_rgba(239,68,68,0.5)]'
      }`}
      title={online ? 'Backend online' : 'Backend offline'}
    />
  )
}

function AppShell() {
  const location = useLocation()
  const isStandalone = location.pathname === '/memory'

  useEffect(() => {
    document.documentElement.classList.add('dark')
  }, [])

  const fallback = (
    <div className="flex items-center justify-center h-screen bg-background">
      <div className="w-5 h-5 border-2 border-zinc-600 border-t-zinc-300 rounded-full animate-spin" />
    </div>
  )

  // Neo-Memory standalone: no nav chrome, fullscreen memory
  if (isStandalone) {
    return (
      <div className="h-screen bg-background">
        <Suspense fallback={fallback}>
          <MemoryPage />
        </Suspense>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b">
        <div className="container mx-auto flex h-14 items-center gap-4 px-6">
          <HealthDot />
          <span className="text-lg font-bold tracking-tight">Alchemy</span>
          <nav className="flex items-center gap-1">
            <NavItem to="/">Dashboard</NavItem>
            <NavItem to="/brain-physics">BrainPhysics</NavItem>
            <NavItem to="/memory">Memory</NavItem>
            <NavItem to="/settings">Settings</NavItem>
          </nav>
        </div>
      </header>

      <main className="container mx-auto px-6 py-8">
        <Suspense fallback={fallback}>
          <Routes>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/brain-physics" element={<BrainPhysicsPage />} />
            <Route path="/memory" element={<MemoryPage />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Routes>
        </Suspense>
      </main>
    </div>
  )
}

export default function App() {
  return (
    <HashRouter>
      <TooltipProvider>
        <AppShell />
      </TooltipProvider>
    </HashRouter>
  )
}
