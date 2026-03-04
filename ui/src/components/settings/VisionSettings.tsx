import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { Input } from '@/components/ui/input'
import { Switch } from '@/components/ui/switch'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'

export function VisionSettings() {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Vision Agent (Tier 2)</CardTitle>
          <CardDescription>
            UI-TARS screenshot-based agent for native Win32 / shadow desktop
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="v-model">CPU Model (72B)</Label>
              <Select defaultValue="rashakol/UI-TARS-72B-DPO">
                <SelectTrigger id="v-model">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="rashakol/UI-TARS-72B-DPO">UI-TARS 72B DPO</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="v-fast">Fast Model (7B)</Label>
              <Select defaultValue="UI-TARS-1.5-7B">
                <SelectTrigger id="v-fast">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="UI-TARS-1.5-7B">UI-TARS 7B</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="v-steps">Max Steps</Label>
              <Input id="v-steps" type="number" min="1" max="200" defaultValue={50} />
            </div>

            <div className="space-y-2">
              <Label htmlFor="v-timeout">Timeout (seconds)</Label>
              <Input id="v-timeout" type="number" min="30" max="900" defaultValue={300} />
            </div>

            <div className="space-y-2">
              <Label htmlFor="v-approval-timeout">Approval Timeout (s)</Label>
              <Input id="v-approval-timeout" type="number" min="10" max="300" defaultValue={60} />
            </div>

            <div className="space-y-2">
              <Label htmlFor="v-history">History Window</Label>
              <Input id="v-history" type="number" min="1" max="20" defaultValue={4} />
            </div>
          </div>

          <div className="flex items-center justify-between">
            <Label htmlFor="v-streaming">Streaming</Label>
            <Switch id="v-streaming" defaultChecked={true} />
          </div>

          <div className="flex items-center justify-between">
            <Label htmlFor="v-routing">Smart Model Routing</Label>
            <Switch id="v-routing" defaultChecked={true} />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Screenshot Pipeline</CardTitle>
          <CardDescription>
            Image format, quality, and resize for token efficiency
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="ss-format">Format</Label>
              <Select defaultValue="jpeg">
                <SelectTrigger id="ss-format">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="jpeg">JPEG (smaller)</SelectItem>
                  <SelectItem value="png">PNG (lossless)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="ss-quality">JPEG Quality</Label>
              <Input id="ss-quality" type="number" min="10" max="100" defaultValue={85} />
            </div>

            <div className="space-y-2">
              <Label htmlFor="ss-width">Resize Width</Label>
              <Input id="ss-width" type="number" min="640" max="3840" defaultValue={1280} />
            </div>

            <div className="space-y-2">
              <Label htmlFor="ss-height">Resize Height</Label>
              <Input id="ss-height" type="number" min="360" max="2160" defaultValue={720} />
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
