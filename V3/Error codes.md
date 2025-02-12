the Raspberry Pi is the Gateway.
Errors that start with E0 are from Arduino Mega.
Arduino Mega will first notify the gateway that an error exists then the error code will be sent if the gateway requests it by sending /ERROR_STATUS	command.
Arduino Mega will send /ERROR_EXISTS to the gateway if an error exists.
| Error | Description |
|-----|-----|
| **E0001** | no feedback from X axis encoder |
| E0002 | no feedback from Y axis encoder|
| E0003 | no feedback from R axis encoder|
| E0004 | X axis micro-switch failure|
| E0005 | Y axis micro-switch failure|
| E0006 | unable to communicate with gateway (Raspberry Pi) | user will get notified of this failure by hearing a vibrating noise from all three stepper motors|
| E0007 | X axis encoder phase 1 failure|
| E0008 | X axis encoder phase 2 failure|
| E0009 | Y axis encoder phase 1 failure|
| E0010 | Y axis encoder phase 2 failure|
| E0011 | R axis encoder phase 1 failure|
| E0012 | R axis encoder phase 2 failure|
| E0013 | unexpected data recieved from Gateway|
| E0014 | low voltage |


