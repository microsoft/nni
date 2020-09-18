import * as React from 'react';
import { MessageBar, MessageBarType } from '@fluentui/react';

interface MessageInfoProps {
    info: string;
    typeInfo: string;
    className?: string;
}

class MessageInfo extends React.Component<MessageInfoProps, {}> {
    constructor(props: MessageInfoProps) {
        super(props);
    }

    render(): React.ReactNode {
        const { info, typeInfo, className } = this.props;
        return (
            <MessageBar messageBarType={MessageBarType[typeInfo]} isMultiline={true} className={className}>
                {info}
            </MessageBar>
        );
    }
}

export default MessageInfo;
