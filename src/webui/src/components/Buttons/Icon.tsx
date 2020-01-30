import * as React from 'react';
import { Icon, initializeIcons } from 'office-ui-fabric-react';
initializeIcons();

const infoIcon = <Icon iconName='info' />
const warining = <Icon iconName='Warning' />
const errorBadge = <Icon iconName='ErrorBadge' />
const completed = <Icon iconName='Completed' />
const blocked = <Icon iconName='StatusCircleBlock' />
const copy = <Icon iconName='Copy' />
const tableListIcon = <Icon iconName='BulletedList' />
const downLoadIcon = { iconName: 'Download' };
const infoIconAbout = { iconName: 'info' };
const timeIcon = { iconName: 'ReminderTime' };

export { infoIcon, warining, errorBadge, completed, blocked,
    infoIconAbout, copy, tableListIcon, downLoadIcon, timeIcon
};
