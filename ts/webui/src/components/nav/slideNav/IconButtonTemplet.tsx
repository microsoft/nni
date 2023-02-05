import React from 'react';
import { DefaultButton, Icon } from '@fluentui/react';

interface ButtonProps {
    icon: string;
    btuName: string;
    event: any;
}

const IconButtonTemplet = (props: ButtonProps): any => {
    const { icon, btuName, event } = props;
    return (
        <div className='container'>
            <DefaultButton className='icon'>
                <Icon iconName={icon} />
            </DefaultButton>
            <DefaultButton className='integralBtn' onClick={event}>
                <Icon iconName={icon} />
                <span className='margin'>{btuName}</span>
            </DefaultButton>
        </div>
    );
};

export default IconButtonTemplet;
