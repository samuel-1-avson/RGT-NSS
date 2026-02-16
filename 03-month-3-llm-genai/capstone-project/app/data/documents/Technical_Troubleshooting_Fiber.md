# Fiber Internet Troubleshooting Guide

**Service:** High-Speed Fiber Optic
**Applies To:** Residential & Small Business

## 1. Initial Assessment

Before escalating to a Level 2 Technician, verify the following:

1. **Power Cycle:** Instruct customer to unplug the ONT (Optical Network Terminal) and Router for 30 seconds.
2. **Status Lights:** Ask customer to describe the lights on the ONT.
   - **Solid Green (PON/WAN)**: Connection is good. Issue likely internal (Router/Device).
   - **Solid Red (LOS/Alarm)**: Loss of Signal. Physical fiber break likely. **Action: Schedule Tech.**
   - **Blinking Green**: Attempting to synchronize. Wait 2 minutes.

## 2. Speed Issues (Slow Connection)

**Complaint**: "I'm not getting the speed I pay for."

### 2.1 Wired Test

Always test speed via **Ethernet cable** directly to the router. Wi-Fi speeds are variable and not guaranteed.

- Ask customer to connect a laptop via Ethernet.
- Run speed test at `speed.telco.com`.
- If speed < 80% of plan: **Provisioning Issue.** Check backend profile.
- If speed > 80% of plan: **Wi-Fi Issue.** Proceed to Wi-Fi troubleshooting.

### 2.2 Wi-Fi Troubleshooting

1. **Frequency Band**: Ensure device is on 5GHz band, not 2.4GHz.
2. **Placement**: Router should be central, elevated, and away from obstructions (metal, concrete).
3. **Channel Congestion**: Log into router admin and change channel to "Auto" or manually select least congested (1, 6, 11 for 2.4GHz).

## 3. Intermittent Connection

**Complaint**: "Internet drops randomly."

### 3.1 Check Signal Levels

- **Rx Optical Power**: Must be between -8 dBm and -25 dBm.
- If roughly -27 dBm or lower: **Marginal Signal.** Inspect drop cable for bends/kinks.

### 3.2 Router Logs

- Check logs for "WAN Disconnected" events.
- If frequent disconnects: Consider replacing ONT/Router.
