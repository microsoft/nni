import * as React from 'react';
import { Stack } from '@fluentui/react';
import { TitleContext } from '../Overview';
import { Icon, initializeIcons } from '@fluentui/react';
import '../../static/style/overview/overviewTitle.scss';
initializeIcons();

export const Title = (): any => (
    <TitleContext.Consumer>
        {(value): React.ReactNode => (
            <Stack horizontal className='panelTitle'>
                <Icon iconName={value.icon} />
                <span style={{ color: value.fontColor }}>{value.text}</span>
            </Stack>
        )}
    </TitleContext.Consumer>
);

/**
 * Experiment: TestParameter | AutoRacing
 * Duration: Timer
 * Trial numbers: NumberSymbol
 * Trial metric chart: HomeGroup
 * Top trials: BulletedList  | tableListIcon
 * succeed table button: MarketDown Market
 * 
 * 
 * 
 * Trial detail
 * defatult metric: HomeGroup 
 * hyper-parameter:
 * duration: BarChartHorizontal
 * Intermediate: StackedLineChart
 * table: BulletedList
 */