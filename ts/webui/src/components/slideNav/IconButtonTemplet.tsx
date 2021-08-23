import React from 'react';
import { DefaultButton, Icon } from '@fluentui/react';

interface ButtonProps {
    icon: string;
    btuName: string;
    event: any;
}
class IconButtonTemplet extends React.Component<ButtonProps, {}> {
    constructor(props: ButtonProps) {
        super(props);
    }

    render(): React.ReactNode {
        const { icon, btuName, event } = this.props;
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
    }
}
export default IconButtonTemplet;
