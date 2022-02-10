import React from 'react';
import { Stack, Icon } from '@fluentui/react';
import { TitleContext } from './TitleContext';
import '../../../static/style/experiment/overview/overviewTitle.scss';

export const Title = (): any => (
    <TitleContext.Consumer>
        {(value): React.ReactNode => (
            <Stack horizontal className='panelTitle'>
                <Icon iconName={value.icon} />
                <span>{value.text}</span>
            </Stack>
        )}
    </TitleContext.Consumer>
);
