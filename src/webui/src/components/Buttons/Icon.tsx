import * as React from 'react';
import { Icon, initializeIcons } from 'office-ui-fabric-react';
initializeIcons();

const a = <Icon iconName='info'/>
const warining = <Icon iconName='Warning' />
const errorBadge = <Icon iconName='ErrorBadge' />
const completed = <Icon iconName='Completed' />
const blocked = <Icon iconName='StatusCircleBlock' />
const copy = <Icon iconName='Copy' />

export { a, warining, errorBadge, completed, blocked, copy };
