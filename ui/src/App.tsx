import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import { TooltipProvider } from '@/components/ui/tooltip'
import { DashboardPage } from '@/pages/Dashboard'
import { SettingsPage } from '@/pages/Settings'

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

export default function App() {
  return (
    <BrowserRouter>
      <TooltipProvider>
        <div className="min-h-screen bg-background">
          <header className="border-b">
            <div className="container mx-auto flex h-14 items-center gap-4 px-6">
              <span className="text-lg font-bold tracking-tight">AlchemyFrontDev</span>
              <nav className="flex items-center gap-1">
                <NavItem to="/">Dashboard</NavItem>
                <NavItem to="/settings">Settings</NavItem>
              </nav>
            </div>
          </header>

          <main className="container mx-auto px-6 py-8">
            <Routes>
              <Route path="/" element={<DashboardPage />} />
              <Route path="/settings" element={<SettingsPage />} />
            </Routes>
          </main>
        </div>
      </TooltipProvider>
    </BrowserRouter>
  )
}
