import React, { useContext } from 'react';
import { Stack, Icon } from '@fluentui/react';
import { TitleContext } from './TitleContext';
import '@style/experiment/overview/overviewTitle.scss';

export const Title = (): any => {
    const { icon, text } = useContext(TitleContext);
    return (
        <Stack horizontal className='panelTitle'>
            <Icon iconName={icon} />
            <span className='fontColor333'>{text}</span>
        </Stack>
    );
};
