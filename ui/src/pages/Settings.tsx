import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { PlaywrightSettings } from '@/components/settings/PlaywrightSettings'
import { VisionSettings } from '@/components/settings/VisionSettings'
import { SystemSettings } from '@/components/settings/SystemSettings'

export function SettingsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Settings</h1>
        <p className="text-muted-foreground">Configure Alchemy agent behavior and system parameters</p>
      </div>

      <Tabs defaultValue="playwright" className="space-y-6">
        <TabsList>
          <TabsTrigger value="playwright">Playwright</TabsTrigger>
          <TabsTrigger value="vision">Vision</TabsTrigger>
          <TabsTrigger value="system">System</TabsTrigger>
        </TabsList>

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
  )
}
