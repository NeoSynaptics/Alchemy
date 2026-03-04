import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { Input } from '@/components/ui/input'
import { Switch } from '@/components/ui/switch'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'

export function PlaywrightSettings() {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Playwright Agent (Tier 1)</CardTitle>
          <CardDescription>
            Browser automation via accessibility tree + Qwen3 14B reasoning
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="flex items-center justify-between">
            <Label htmlFor="pw-enabled">Enabled</Label>
            <Switch id="pw-enabled" defaultChecked={true} />
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="pw-model">Model</Label>
              <Select defaultValue="qwen3:14b">
                <SelectTrigger id="pw-model">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="qwen3:14b">Qwen3 14B</SelectItem>
                  <SelectItem value="qwen2.5-coder:14b">Qwen2.5-Coder 14B</SelectItem>
                  <SelectItem value="qwen2.5-coder:32b">Qwen2.5-Coder 32B</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="pw-temp">Temperature</Label>
              <Input id="pw-temp" type="number" step="0.05" min="0" max="2" defaultValue={0.1} />
            </div>

            <div className="space-y-2">
              <Label htmlFor="pw-tokens">Max Tokens</Label>
              <Input id="pw-tokens" type="number" step="128" min="128" max="8192" defaultValue={1024} />
            </div>

            <div className="space-y-2">
              <Label htmlFor="pw-steps">Max Steps</Label>
              <Input id="pw-steps" type="number" min="1" max="200" defaultValue={50} />
            </div>

            <div className="space-y-2">
              <Label htmlFor="pw-settle">Settle Timeout (ms)</Label>
              <Input id="pw-settle" type="number" step="500" min="0" max="30000" defaultValue={5000} />
            </div>

            <div className="space-y-2">
              <Label htmlFor="pw-elements">Max Snapshot Elements</Label>
              <Input id="pw-elements" type="number" min="10" max="500" defaultValue={75} />
            </div>
          </div>

          <div className="flex items-center justify-between">
            <Label htmlFor="pw-think">Think Mode (Qwen3 reasoning)</Label>
            <Switch id="pw-think" defaultChecked={true} />
          </div>

          <div className="flex items-center justify-between">
            <Label htmlFor="pw-headless">Headless Browser</Label>
            <Switch id="pw-headless" defaultChecked={true} />
          </div>

          <div className="flex items-center justify-between">
            <Label htmlFor="pw-approval">Approval Gates</Label>
            <Switch id="pw-approval" defaultChecked={true} />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Escalation (Tier 1.5)</CardTitle>
          <CardDescription>
            Fallback to UI-TARS 7B when Playwright agent gets stuck
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="flex items-center justify-between">
            <Label htmlFor="esc-enabled">Escalation Enabled</Label>
            <Switch id="esc-enabled" defaultChecked={true} />
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="esc-model">Escalation Model</Label>
              <Select defaultValue="UI-TARS-1.5-7B">
                <SelectTrigger id="esc-model">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="UI-TARS-1.5-7B">UI-TARS 7B</SelectItem>
                  <SelectItem value="rashakol/UI-TARS-72B-DPO">UI-TARS 72B</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="esc-parse">Parse Failures Threshold</Label>
              <Input id="esc-parse" type="number" min="1" max="10" defaultValue={3} />
            </div>

            <div className="space-y-2">
              <Label htmlFor="esc-repeat">Repeated Actions Threshold</Label>
              <Input id="esc-repeat" type="number" min="1" max="10" defaultValue={3} />
            </div>

            <div className="space-y-2">
              <Label htmlFor="esc-complex">Complexity Threshold (refs)</Label>
              <Input id="esc-complex" type="number" min="10" max="500" defaultValue={60} />
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
