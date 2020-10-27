import * as React from 'react';
import { Icon, initializeIcons } from '@fluentui/react';
initializeIcons();

const infoIcon = <Icon iconName='info' />;
const warining = <Icon iconName='Warning' />;
const errorBadge = <Icon iconName='ErrorBadge' />;
const completed = <Icon iconName='Completed' />;
const blocked = <Icon iconName='StatusCircleBlock' />;
const copy = <Icon iconName='Copy' />;
const tableListIcon = <Icon iconName='BulletedList' />;
const downLoadIcon = { iconName: 'Download' };
const infoIconAbout = { iconName: 'info' };
const timeIcon = { iconName: 'ReminderTime' };
const disableUpdates = { iconName: 'DisableUpdates' };
const requency = { iconName: 'Timer' };
const closeTimer = { iconName: 'Blocked2' };
const LineChart = <Icon iconName='LineChart' />;
const Edit = <Icon iconName='Edit' />;
const CheckMark = <Icon iconName='CheckMark' />;
const Cancel = <Icon iconName='Cancel' />;

export {
    infoIcon,
    warining,
    errorBadge,
    completed,
    blocked,
    infoIconAbout,
    copy,
    tableListIcon,
    downLoadIcon,
    timeIcon,
    disableUpdates,
    requency,
    closeTimer,
    LineChart,
    Edit,
    CheckMark,
    Cancel
};
