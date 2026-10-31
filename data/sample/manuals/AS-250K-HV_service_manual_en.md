---
title: AS-250K-HV String Inverter Service Manual
product_models: [AS-250K-HV]
doc_type: manual
version: V2.0
---

# AS-250K-HV String Inverter Service Manual

> Sample data: ACME Solar is a fictional brand. All content, parameters and thresholds in this manual are made up for demonstrating the RAG framework and must not be used on real equipment.

## 1 Safety

The AS-250K-HV operates at up to 1500 V DC. Lethal voltages are present on the DC input terminals whenever the PV array is illuminated, even when the DC switch is OFF.

Before opening the enclosure: open the AC breaker; turn the DC switch to OFF; disconnect all PV strings; wait at least 10 minutes for the DC bus capacitors to discharge; verify with a multimeter that the DC bus voltage is below 36 V.

## 2 Fault codes

| Code | Name | Type | Action |
| --- | --- | --- | --- |
| E037 | Low insulation resistance | Fault | Check insulation of each string to ground, see 3.1 |
| E051 | Fan fault | Fault | Check external fan 310-04522-A and its connector |
| E063 | Over-temperature | Fault | Check fans, air filters and installation clearance, see 3.2 |
| W102 | String current mismatch | Warning | Check shading, soiling and string configuration |

## 3 Troubleshooting

### 3.1 E037 Low insulation resistance

The inverter measures PV-to-ground insulation resistance before connecting to the grid. For 1500 V systems the threshold is 100 kΩ; below this value E037 is raised and the inverter will not connect.

Measure each string with a 1500 V insulation tester (positive-to-ground and negative-to-ground). Values above 1 MΩ are normal. Replace damaged cables or MC4 connectors 610-00045-A on the faulty string.

### 3.2 E063 Over-temperature

E063 is raised when the internal heat-sink temperature exceeds 85 °C. Above an ambient temperature of 45 °C the inverter derates output power automatically, which is normal behavior and not a fault.

Typical causes: clogged air filters, failed external fan 310-04522-A, insufficient clearance (keep at least 500 mm above and 300 mm on each side), direct afternoon sun on the heat sink.

Steps: clean or replace the air filters; check that all external fans spin and that there is no E051; verify clearances; consider installing a sun shade.

## 4 Component replacement

### 4.1 Replacing the DC switch

Use DC switch 220-11009-A (1500 V rated) only. Never install the 1100 V DC switch 220-11008-A in an AS-250K-HV: it is not rated for 1500 V systems and may arc or fail.

Tighten the DC switch terminal screws to 4.0 N·m.
