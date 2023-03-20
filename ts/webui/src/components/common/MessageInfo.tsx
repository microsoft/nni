import * as React from 'react';
import { MessageBar, MessageBarType } from '@fluentui/react';

interface MessageInfoProps {
    info: string;
    typeInfo: string;
    className?: string;
}

const MessageInfo = (props: MessageInfoProps): any => {
    const { info, typeInfo, className } = props;

    return (
        <MessageBar messageBarType={MessageBarType[typeInfo]} isMultiline={true} className={className}>
            {info}
        </MessageBar>
    );
};

export default MessageInfo;
