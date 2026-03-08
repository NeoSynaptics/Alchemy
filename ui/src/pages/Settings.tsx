import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { PlaywrightSettings } from '@/components/settings/PlaywrightSettings'
import { VisionSettings } from '@/components/settings/VisionSettings'
import { SystemSettings } from '@/components/settings/SystemSettings'
import { VoiceSettings } from '@/components/settings/VoiceSettings'
import { ChatPanel } from '@/components/ChatPanel'

export function SettingsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Settings</h1>
        <p className="text-muted-foreground">Configure Alchemy agent behavior and system parameters</p>
      </div>

      <div className="flex gap-6">
        {/* Settings tabs */}
        <div className="min-w-0 flex-1">
          <Tabs defaultValue="voice" className="space-y-6">
            <TabsList>
              <TabsTrigger value="voice">Voice</TabsTrigger>
              <TabsTrigger value="playwright">Playwright</TabsTrigger>
              <TabsTrigger value="vision">Vision</TabsTrigger>
              <TabsTrigger value="system">System</TabsTrigger>
            </TabsList>

            <TabsContent value="voice">
              <VoiceSettings />
            </TabsContent>

            <TabsContent value="playwright">
              <PlaywrightSettings />
            </TabsContent>

            <TabsContent value="vision">
              <VisionSettings />
            </TabsContent>

            <TabsContent value="system">
              <SystemSettings />
            </TabsContent>
          </Tabs>
        </div>

        {/* Chat panel — visible on large screens */}
        <div className="hidden w-[380px] shrink-0 lg:block" style={{ height: 'calc(100vh - 12rem)' }}>
          <ChatPanel />
        </div>
      </div>
    </div>
  )
}
